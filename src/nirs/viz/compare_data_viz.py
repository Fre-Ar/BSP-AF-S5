# src/nirs/viz/compare_data_viz.py

# --- I/O ---
import os
import tempfile
from pathlib import Path
from contextlib import contextmanager

# --- Math ---
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Optional

# --- Our modules ---
from .visualizer import plot_geopandas
from nirs.engine import Predictor
from nirs.inference import InferenceConfig
from nirs.metrics import compute_classification_metrics, compute_distance_metrics
from nirs.training import aggregate_params
from utils.utils_geo import METRICS_CSV, DATA_ANALYSIS_PATH


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
    
    plt.tight_layout()
    plt.show()


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
# Public: comparison / visualization entry points
# -------------------------------------------------------------------

def visualize_model(
    parquet_path: str,
    checkpoint_path: str,
    
    # MLP params
    model_cfg: InferenceConfig,
    
    # Data params
    sample: int | None = 200_000,
    batch_size: int = 131_072,
    device: str | None = None,
    
    predictions_only: bool = False,
    show_plots: bool = True,
    overrides=None,
    
    out_dir: str = DATA_ANALYSIS_PATH,
    metrics_csv: str = METRICS_CSV
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
    # 1. Load subset of columns from parquet
    cols = ["lon", "lat", "x", "y", "z", "dist_km", "log1p_dist", "c1_id", "c2_id"]
    df = pd.read_parquet(parquet_path, columns=cols)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=0).reset_index(drop=True)
        
    # Normalize XYZ to unit vectors
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32, copy=True)
    nrm = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = (xyz / np.clip(nrm, 1e-9, None)).astype(np.float32)

    
    # 2. Run inference and get predictions
    predictor = Predictor(model_cfg, checkpoint_path, device)
    
    N = len(df)
    pred_dist = np.zeros(N, dtype=np.float32)
    pred_c1 = np.zeros(N, dtype=np.int64)
    pred_c2 = np.zeros(N, dtype=np.int64)
    
    # We collect logits in lists
    all_logits_c1 = []
    all_logits_c2 = []

    print(f"[Compare] Running inference on {N} points...")
    for s_idx in range(0, N, batch_size):
        e_idx = min(N, s_idx + batch_size)
        batch_xyz = xyz[s_idx:e_idx]
        
        # We need logits for metrics
        p = predictor.predict(batch_xyz, return_logits=True)
        pred_dist[s_idx:e_idx] = p.dist_km
        pred_c1[s_idx:e_idx] = p.c1_ids
        pred_c2[s_idx:e_idx] = p.c2_ids
        
        if not predictions_only:
            all_logits_c1.append(p.logits_c1)
            all_logits_c2.append(p.logits_c2)
    
    # 3. Visualizations
    if predictions_only:
        # (A) predicted distance (km)
        _plot_derived_field(df, pred_dist, "pred_dist_km", color_mode="continuous", log_scale=True, overrides=overrides)
        # (B) and (C) predicted c1 and c2 (hashed colors)
        for name, arr in [("pred_c1", pred_c1), ("pred_c2", pred_c2)]:
            _plot_derived_field(df,arr,name,color_mode="hashed",markersize=3,overrides=overrides)
        return {"pred_dist": pred_dist, "pred_c1": pred_c1, "pred_c2": pred_c2}

    # 4. Metrics & Comparison
    print("[Compare] Computing metrics...")
    
    # Prepare Ground Truth
    y_dist_km = df["dist_km"].to_numpy(dtype=np.float32)
    y_c1 = df["c1_id"].to_numpy(dtype=np.int64)
    y_c2 = df["c2_id"].to_numpy(dtype=np.int64)

    # Basic Visuals
    err_km = np.abs(pred_dist - y_dist_km)
    c1_ok = (pred_c1 == y_c1)
    c2_ok = (pred_c2 == y_c2)

    if show_plots:
        _plot_derived_field(df, err_km, "err_km", color_mode="continuous", log_scale=True, overrides=overrides)
        _plot_green_red(df["lon"].values, df["lat"].values, c1_ok, "c1 Correctness")
        _plot_green_red(df["lon"].values, df["lat"].values, c2_ok, "c2 Correctness")

    # Full Metrics Calculation
    stats = {}
    
    # Convert collected numpy arrays back to Torch CPU tensors
    t_pred_dist = torch.from_numpy(pred_dist)
    t_y_dist = torch.from_numpy(y_dist_km)
    
    t_logits_c1 = torch.from_numpy(np.concatenate(all_logits_c1, axis=0))
    t_y_c1_id = torch.from_numpy(y_c1)
    
    t_logits_c2 = torch.from_numpy(np.concatenate(all_logits_c2, axis=0))
    t_y_c2_id = torch.from_numpy(y_c2)
    # A. Distance Metrics
    d_metrics = compute_distance_metrics(t_pred_dist, t_y_dist)
    stats.update(d_metrics)
    
    # B. Classification Metrics
    # We need to reconstruct the bits tensor if ECOC
    if model_cfg.label_mode == "ecoc":
        if predictor.codebook is None:
            raise ValueError("label_mode='ecoc' requires a codebook.")
        c1_bits = np.stack([predictor.codebook[int(cid)] for cid in y_c1], axis=0).astype(np.float32)
        c2_bits = np.stack([predictor.codebook[int(cid)] for cid in y_c1], axis=0).astype(np.float32)
        c1_bits = torch.from_numpy(c1_bits)
        c2_bits = torch.from_numpy(c2_bits)
    else:
        c1_bits = None
        c2_bits = None
    
    cpu_codes = predictor.codes_mat.cpu() if predictor.codes_mat is not None else None
    cpu_ids = predictor.class_ids.cpu() if predictor.class_ids is not None else None
    cpu_pw1 = predictor.pw_c1.cpu() if predictor.pw_c1 is not None else None
    cpu_pw2 = predictor.pw_c2.cpu() if predictor.pw_c2 is not None else None
    
    c1_metrics = compute_classification_metrics(
        logits=t_logits_c1,
        targets=t_y_c1_id,
        target_dist_km=t_y_dist,
        label_mode=model_cfg.label_mode,
        codes_mat=cpu_codes,
        class_ids=cpu_ids,
        pos_weight=cpu_pw1,
        bits=c1_bits
    )
    for k, v in c1_metrics.items(): stats[f"c1_{k}"] = v
    
    # Compute C2
    c2_metrics = compute_classification_metrics(
        logits=t_logits_c2,
        targets=t_y_c2_id,
        target_dist_km=t_y_dist,
        label_mode=model_cfg.label_mode,
        codes_mat=cpu_codes,
        class_ids=cpu_ids,
        pos_weight=cpu_pw2,
        bits=c2_bits
    )
    for k, v in c2_metrics.items(): stats[f"c2_{k}"] = v
    
    num_params = sum(p.numel() for p in predictor.model.parameters())
    ckpt_cfg = predictor.ckpt.get("config", {})
    epochs_trained = predictor.ckpt.get("epoch", 0)
    
    global_meta = aggregate_params(
        model_cfg, 
        num_params, 
        lr=ckpt_cfg['lr'], 
        weight_decay=ckpt_cfg['wd'], 
        train_set=ckpt_cfg['train_set'], 
        eval_set=parquet_path,
        epochs_trained=epochs_trained)
    
    row = [{
        **global_meta,
        **stats
    }]
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / metrics_csv
    
    df = pd.DataFrame(row)
    df.to_csv(
        csv_path,
        mode="a",                     # append
        header=not csv_path.exists(), # write header only once
        index=False
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