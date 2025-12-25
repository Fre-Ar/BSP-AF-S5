# src/nirs/viz/compare_data_viz.py

# --- I/O ---
import os
from dataclasses import dataclass
from contextlib import contextmanager
import tempfile

# --- Math ---
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
import geopandas as gpd
import torch.nn.functional as F

# --- Our module ---
from .visualizer import plot_geopandas, _prep_dataframe
from nirs.create_nirs import build_model
from geodata.ecoc.ecoc import (
    load_ecoc_codes,
    ecoc_decode,
    _prepare_codebook_tensor
)

from nirs.engine import (
    compute_potential_ecoc_pos_weights,
    load_model_and_codebook,
    LabelMode)


# -------------------------------------------------------------------
# Learnable hyperparameter inspection (INCODE a,b,c,d sanity check)
# -------------------------------------------------------------------
def summarize_abcd(a, b, c, d, label=""):
    """
    Prints simple summary stats (mean/std/min/max) for four tensors a,b,c,d.

    Useful for checking that INCODE hyperparameters don't explode or collapse.
    """
    def s(t):
        return dict(mean=float(t.mean()), std=float(t.std()),
                    min=float(t.min()), max=float(t.max()))
    print(f"[{label}] a={s(a)}")
    print(f"[{label}] b={s(b)}")
    print(f"[{label}] c={s(c)}")
    print(f"[{label}] d={s(d)}")

@torch.no_grad()
def check_abcd(model, x_batch):
    """
    Runs a single forward pass with `return_abcd=True` and print stats for (a,b,c,d).

    Assumes model(x, return_abcd=True) -> (dist, c1, c2, (a,b,c,d)).
    """
    model.eval()
    if x_batch.dim() == 1: 
        x_batch = x_batch.unsqueeze(0)
    dist, c1, c2, (a,b,c,d) = model(x_batch, return_abcd=True)
    summarize_abcd(a, b, c, d, label=f"B={x_batch.size(0)}")
    return a, b, c, d

# -------------------------------------------------------------------
# Small plotting helper (green/red correctness scatter)
# -------------------------------------------------------------------
def _plot_green_red(lon, lat, ok_mask, title, figsize=(11,5), s=3, alpha=0.9):
    """
    Quick scatter on a world basemap highlighting correct vs wrong predictions.

    Green  = correct
    Red    = wrong
    """
    fig, ax = plt.subplots(figsize=figsize)
    try:
        from geodatasets import get_path
        world = gpd.read_file(get_path("naturalearth.land"))
        world.plot(ax=ax, color='whitesmoke', edgecolor='gray', linewidth=0.3, zorder=1)
    except Exception:
        # Basemap is optional; fall back to just scatter.
        pass
    
    ok = ok_mask.astype(bool)
    if ok.any():
        ax.scatter(lon[ok],  lat[ok],  s=s, c='green', alpha=alpha, edgecolors='none', zorder=3, label='correct')
    if (~ok).any():
        ax.scatter(lon[~ok], lat[~ok], s=s, c='red',   alpha=alpha, edgecolors='none', zorder=3, label='wrong')
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.legend(loc="lower left")
    
    plt.tight_layout(); plt.show()


# -------------------------------------------------------------------
# Plotting helpers
# -------------------------------------------------------------------

@contextmanager
def _temp_parquet_from_df(df: pd.DataFrame):
    """
    Write a DataFrame to a temporary Parquet file, yield its path, and
    delete the file afterwards.
    """
    fd, path = tempfile.mkstemp(suffix=".parquet", prefix="nir_viz_")
    os.close(fd)
    try:
        df.to_parquet(path, index=False)
        yield path
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def _plot_derived_field(
    df: pd.DataFrame,
    values: np.ndarray,
    value_col: str,
    *,
    color_mode: str = "continuous",
    log_scale: bool = False,
    markersize: int = 2,
    overrides=None,
    clip_quantiles: tuple[float, float] = (0.01, 0.99),
    figsize=(11, 5),
):
    """
    Convenience wrapper:
      - takes an existing df with 'lon' and 'lat'
      - attaches a derived column (value_col := values)
      - writes a temporary Parquet
      - calls plot_geopandas()
      - cleans up the temporary file

    Assumes len(values) == len(df).
    """
    tmp = df[["lon", "lat"]].copy()
    tmp[value_col] = values

    with _temp_parquet_from_df(tmp) as tmp_path:
        plot_geopandas(
            tmp_path,
            lon="lon",
            lat="lat",
            color_by=value_col,
            color_mode=color_mode,
            log_scale=log_scale,
            clip_quantiles=clip_quantiles,
            sample=None,
            markersize=markersize,
            alpha=0.9,
            figsize=figsize,
            overrides=overrides,
        )

# -------------------------------------------------------------------
# Internal helper: inference wrapper
# -------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Bundle of arrays returned by _run_model_on_parquet."""
    df: pd.DataFrame           # sampled dataframe with lon/lat/x/y/z/dist_km/...
    pred_dist: np.ndarray      # predicted distance in km (N,)
    pred_c1: np.ndarray        # predicted c1 id (N,)
    pred_c2: np.ndarray        # predicted c2 id (N,)

@torch.no_grad()
def _run_model_on_parquet(
    parquet_path: str,
    checkpoint_path: str,
    model_name: str,
    
    layer_counts,
    w0: float,
    w_h: float,
    s_param: float,
    beta: float,
    global_z: bool,
    regularize_hyperparams: bool,
    
    label_mode: LabelMode = "ecoc",
    codes_path: str | None = None,
    sample: int | None = 200_000,
    batch_size: int = 131_072,
    
    device: str | None = None,
    model_outputs_log1p: bool = True,
) -> InferenceResult:
    """
    Core helper: loads a parquet, restores the model from checkpoint,
    run batched inference, and return predictions and the sampled dataframe.

    - Handles ECOC vs softmax.
    - Normalizes xyz to unit vectors.
    - If ECOC, uses soft ECOC decoding consistent with training pos_weight.
    """
    # load model and codebook from checkpoint
    model, device, codebook, ckpt = load_model_and_codebook(
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        layer_counts=layer_counts,
        
        w0=w0,
        w_hidden=w_h,
        s_param=s_param,
        beta=beta,
        global_z=global_z,
        
        label_mode=label_mode,
        codes_path=codes_path,
        device=device)

    pw_c1, pw_c2 = compute_potential_ecoc_pos_weights(
        parquet_path=parquet_path,
        codebook=codebook,
        label_mode=label_mode)
    
        
    # ---- load subset of columns from parquet ----
    cols = ["lon", "lat", "x", "y", "z", "dist_km", "log1p_dist", "c1_id", "c2_id"]
    df = pd.read_parquet(parquet_path, columns=cols)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=0).reset_index(drop=True)

    # Normalize xyz to unit vectors
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
    nrm = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = (xyz / np.clip(nrm, 1e-9, None)).astype(np.float32)


    # ---- batched inference ----
    N = xyz.shape[0]
    pred_dist = np.zeros(N, dtype=np.float32)
    pred_c1 = np.zeros(N, dtype=np.int64)
    pred_c2 = np.zeros(N, dtype=np.int64)
    
    class_ids, codes_mat = _prepare_codebook_tensor(codebook, device, pred_c1.dtype)  # codes: [C,K]

    for s_idx in range(0, N, batch_size):
        e_idx = min(N, s_idx + batch_size)
        u = torch.from_numpy(xyz[s_idx:e_idx]).to(device)

        # Optional hyperparam inspection for INCODE
        if model_name.lower() == "incode" and s_idx == 0:
            check_abcd(model, u)

        d_hat, logits_c1, logits_c2 = model(u)
        d_hat = d_hat.squeeze(-1)

        if model_outputs_log1p:
            d_km = torch.expm1(d_hat) # convert log1p(dist_km) -> dist_km
        else:
            d_km = d_hat  # already in km

        if label_mode == "ecoc":
            # Use shared ecoc_decode (soft mode) consistent with pos_weight
            c1_idx_pred = ecoc_decode(logits_c1, codes_mat, class_ids, pos_weight=pw_c1, mode="soft")
            c2_idx_pred = ecoc_decode(logits_c2, codes_mat, class_ids, pos_weight=pw_c2, mode="soft")
            pred_c1[s_idx:e_idx] = c1_idx_pred.long().cpu().numpy()
            pred_c2[s_idx:e_idx] = c2_idx_pred.long().cpu().numpy()
        else:
            # Simple softmax argmax
            pred_c1[s_idx:e_idx] = logits_c1.argmax(dim=1).long().cpu().numpy()
            pred_c2[s_idx:e_idx] = logits_c2.argmax(dim=1).long().cpu().numpy()

        pred_dist[s_idx:e_idx] = d_km.float().cpu().numpy()

    return InferenceResult(
        df=df,
        pred_dist=pred_dist,
        pred_c1=pred_c1,
        pred_c2=pred_c2,
    )
    
# -------------------------------------------------------------------
# Public: comparison / visualization entry points
# -------------------------------------------------------------------

def compare_parquet_and_model_ecoc(
    parquet_path: str,
    checkpoint_path: str,
    model_name: str,
    
    # MLP params
    layer_counts,
    depth: int, layer: int,
    w0: float, w_h: float, s_param: float, beta: float,
    global_z: bool,
    regularize_hyperparams: bool,
    
    # Data params
    label_mode: str = "auto",
    codes_path: str | None = None,
    sample: int | None = 200_000,
    batch_size: int = 131_072,
    device: str | None = None,
    model_outputs_log1p: bool = True,
    
    predictions_only: bool = False,
    overrides=None
):
    """
    Compares a trained model against a sampled subset of the training parquet.

    - Loads model from `checkpoint_path`.
    - Runs predictions on points from `parquet_path`.
    - Either:
        * predictions_only=True  → just plot model outputs (dist, c1, c2)
        * predictions_only=False → also compare against ground-truth and plot
                                    error + green/red correctness maps.
    """
    # Run inference and get predictions
    result = _run_model_on_parquet(
        parquet_path=parquet_path,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        layer_counts=layer_counts,
        w0=w0,
        w_h=w_h,
        s_param=s_param,
        beta=beta,
        global_z=global_z,
        regularize_hyperparams=regularize_hyperparams,
        label_mode=label_mode,
        codes_path=codes_path,
        sample=sample,
        batch_size=batch_size,
        device=device,
        model_outputs_log1p=model_outputs_log1p,
    )

    df = result.df
    pred_dist = result.pred_dist
    pred_c1 = result.pred_c1
    pred_c2 = result.pred_c2

    # ===== PREDICTIONS-ONLY MODE =====
    if predictions_only:
        # (A) predicted distance (km)
        _plot_derived_field(
            df,
            pred_dist,
            "pred_dist_km",
            color_mode="continuous",
            log_scale=True,
            markersize=2,
            overrides=overrides,
            clip_quantiles=(0.01, 0.99),
            figsize=(11, 5),
        )
        
        # (B) and (C) predicted c1 and c2 (hashed colors)
        for name, arr in [("pred_c1", pred_c1), ("pred_c2", pred_c2)]:
            _plot_derived_field(
                df,
                arr,
                name,
                color_mode="hashed",
                log_scale=False,
                markersize=3,
                overrides=overrides,
                figsize=(11, 5),
            )

        return {
            "pred_dist": pred_dist,
            "pred_c1": pred_c1,
            "pred_c2": pred_c2,
        }

    # ===== COMPARISON MODE =====
    # Targets
    y_dist_km = df["dist_km"].to_numpy(dtype=np.float32)
    y_log1p_dist = df["log1p_dist"].to_numpy(dtype=np.float32)
    y_c1_id = df["c1_id"].to_numpy(dtype=np.int64)
    y_c2_id = df["c2_id"].to_numpy(dtype=np.int64)

    err_km = np.abs(pred_dist - y_dist_km)
    c1_ok = (pred_c1 == y_c1_id)
    c2_ok = (pred_c2 == y_c2_id)

    # (A) distance error heatmap
    _plot_derived_field(
        df,
        err_km,
        "err_km",
        color_mode="continuous",
        log_scale=True,
        markersize=2,
        overrides=overrides,
        clip_quantiles=(0.01, 0.99),
        figsize=(11, 5),
    )

    # (B) c1 correctness (green/red)
    lon_np = df["lon"].to_numpy(dtype=np.float32)
    lat_np = df["lat"].to_numpy(dtype=np.float32)
    _plot_green_red(
        lon_np, lat_np, c1_ok,
        title="c1 correctness (green=correct, red=wrong)",
    )

    # (C) c2 correctness (green/red)
    _plot_green_red(
        lon_np, lat_np, c2_ok,
        title="c2 correctness (green=correct, red=wrong)",
    )

    print(
        f"[distance]  MAE={err_km.mean():.3f} km | "
        f"median={np.median(err_km):.3f} km | "
        f"95p={np.quantile(err_km, 0.95):.3f} km"
    )
    print(f"[c1]       acc={c1_ok.mean():.4f}")
    print(f"[c2]       acc={c2_ok.mean():.4f}")

    return {
        "err_km": err_km,
        "c1_ok": c1_ok,
        "c2_ok": c2_ok,
        "pred_c1": pred_c1,
        "pred_c2": pred_c2,
        "pred_dist": pred_dist,
    }