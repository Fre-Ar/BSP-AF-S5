import json, math, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from visualizer import plot_geopandas
from siren import SIRENLayer, SIREN
from nir import NIRLayer, NIRTrunk, MultiHeadNIR
from data import BordersParquet, LossWeights, train_one_epoch, evaluate, load_ecoc_codes
import matplotlib.pyplot as plt
import geopandas as gpd

def codebook_to_bits_matrix(codebook: dict[int, np.ndarray], n_bits: int | None = None):
    """
    From {class_id: np.uint8[K]} build:
      ids:  [C] int64 array of class ids (sorted ascending)
      bits: [C,K] float tensor (0/1), where K = n_bits or inferred from entries.
    """
    ids = np.array(sorted(codebook.keys()), dtype=np.int64)
    # infer K from the first entry unless n_bits is provided
    K_inf = int(next(iter(codebook.values())).shape[0])
    K = n_bits if n_bits is not None else K_inf
    M = np.zeros((len(ids), K), dtype=np.uint8)
    for i, cid in enumerate(ids):
        v = codebook[cid]
        if v.shape[0] < K:
            raise ValueError(f"Code for class {cid} has length {v.shape[0]} < requested {K}")
        M[i, :] = v[:K].astype(np.uint8)
    bits = torch.from_numpy(M.astype(np.float32))
    return ids, bits  # ids: np.int64[C], bits: torch.FloatTensor[C,K]

def soft_ecoc_argmax(logits: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
    """
    logits: [B,K] model logits (pre-sigmoid)
    bits:   [C,K] 0/1 codes (float, same device)
    Returns: [B] class index (0..C-1) maximizing soft log-likelihood.
    """
    z = logits.unsqueeze(1)  # [B,1,K]
    b = bits.unsqueeze(0)    # [1,C,K]
    score = b * torch.nn.functional.logsigmoid(z) + (1 - b) * torch.nn.functional.logsigmoid(-z)
    return score.sum(dim=-1).argmax(dim=1)

# ---------- green/red quick scatter ----------
def _plot_green_red(lon, lat, ok_mask, title, figsize=(11,5), s=3, alpha=0.9):
    fig, ax = plt.subplots(figsize=figsize)
    try:
        from geodatasets import get_path
        world = gpd.read_file(get_path("naturalearth.land"))
        world.plot(ax=ax, color='whitesmoke', edgecolor='gray', linewidth=0.3, zorder=1)
    except Exception:
        pass
    ok = ok_mask.astype(bool)
    if ok.any():
        ax.scatter(lon[ok],  lat[ok],  s=s, c='green', alpha=alpha, edgecolors='none', zorder=3, label='correct')
    if (~ok).any():
        ax.scatter(lon[~ok], lat[~ok], s=s, c='red',   alpha=alpha, edgecolors='none', zorder=3, label='wrong')
    ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title); ax.legend(loc="lower left")
    plt.tight_layout(); plt.show()

# ---------- main comparison (updated) ----------
def compare_parquet_and_model_ecoc(
    parquet_path: str,
    checkpoint_path: str,
    codes_path: str,
    model_builder,                 # callable that rebuilds SAME arch as training
    sample: int | None = 200_000,
    batch_size: int = 131_072,
    device: str | None = None,
    n_bits: int | None = None,     # if None, inferred from codebook
    model_outputs_km: bool = True,
    earth_radius_km: float = 6371.0,
    predictions_only: bool = False # <-- NEW: just show model predictions
):
    # device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else: device = "cpu"

    # load parquet
    cols = ["lon","lat","x","y","z","dist_km","c1_id","c2_id"]
    df = pd.read_parquet(parquet_path, columns=cols)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=0).reset_index(drop=True)

    xyz = df[["x","y","z"]].to_numpy(dtype=np.float32, copy=True)
    nrm = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = (xyz / np.clip(nrm, 1e-9, None)).astype(np.float32)

    y_dist_km = df["dist_km"].to_numpy(dtype=np.float32)
    y_c1_id   = df["c1_id"].to_numpy(dtype=np.int64)
    y_c2_id   = df["c2_id"].to_numpy(dtype=np.int64)

    # load codebook (id -> bits np.uint8[K])
    codebook = load_ecoc_codes(codes_path)
    # infer K (or use n_bits)
    K_inf = int(next(iter(codebook.values())).shape[0])
    K = n_bits if n_bits is not None else K_inf

    # build [C,K] bits and [C] ids
    all_ids_np = np.array(sorted(codebook.keys()), dtype=np.int64)
    C = all_ids_np.shape[0]
    M = np.zeros((C, K), dtype=np.uint8)
    for i, cid in enumerate(all_ids_np):
        v = codebook[cid]
        if v.shape[0] < K:
            raise ValueError(f"Code for class {cid} has length {v.shape[0]} < requested {K}")
        M[i, :] = v[:K].astype(np.uint8)
    bits = torch.from_numpy(M.astype(np.float32))
    all_ids = torch.from_numpy(all_ids_np)

    # checkpoint & model
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = model_builder().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # inference
    N = xyz.shape[0]
    pred_dist = np.zeros(N, dtype=np.float32)
    pred_c1   = np.zeros(N, dtype=np.int64)
    pred_c2   = np.zeros(N, dtype=np.int64)

    with torch.no_grad():
        bits = bits.to(device)
        all_ids = all_ids.to(device)
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            u = torch.from_numpy(xyz[s:e]).to(device)
            out = model(u)
            if not (isinstance(out, (tuple, list)) and len(out) >= 3):
                raise RuntimeError("Model must return (distance, logits_c1_bits, logits_c2_bits).")
            d_hat, logits_c1, logits_c2 = out[:3]
            d_hat = d_hat.squeeze(-1)

            if not model_outputs_km:
                d_hat = d_hat * earth_radius_km

            # soft ECOC decoding
            z1 = logits_c1.unsqueeze(1)  # [B,1,K]
            z2 = logits_c2.unsqueeze(1)
            b  = bits.unsqueeze(0)       # [1,C,K]

            s1 = b * torch.nn.functional.logsigmoid(z1) + (1 - b) * torch.nn.functional.logsigmoid(-z1)
            s2 = b * torch.nn.functional.logsigmoid(z2) + (1 - b) * torch.nn.functional.logsigmoid(-z2)

            c1_idx = s1.sum(dim=-1).argmax(dim=1)  # [B]
            c2_idx = s2.sum(dim=-1).argmax(dim=1)  # [B]

            pred_dist[s:e] = d_hat.float().cpu().numpy()
            pred_c1[s:e]   = all_ids[c1_idx].cpu().numpy()
            pred_c2[s:e]   = all_ids[c2_idx].cpu().numpy()

    # === PREDICTIONS-ONLY MODE ===
    if predictions_only:
        # (A) predicted distance (km)
        tmp_pd = df[["lon","lat"]].copy()
        tmp_pd["pred_dist_km"] = pred_dist
        tmp_pd.to_parquet("_tmp_pred_dist.parquet", index=False)
        plot_geopandas("_tmp_pred_dist.parquet",
                       lon="lon", lat="lat",
                       color_by="pred_dist_km",
                       color_mode="continuous",
                       log_scale=True,
                       clip_quantiles=(0.01,0.99),
                       sample=None, markersize=2, alpha=0.9, figsize=(11,5))

        # (B) predicted c1 (hashed colors)
        tmp_c1 = df[["lon","lat"]].copy()
        tmp_c1["pred_c1"] = pred_c1
        tmp_c1.to_parquet("_tmp_pred_c1.parquet", index=False)
        plot_geopandas("_tmp_pred_c1.parquet",
                       lon="lon", lat="lat",
                       color_by="pred_c1",
                       color_mode="hashed",
                       sample=None, markersize=3, alpha=0.9, figsize=(11,5))

        # (C) predicted c2 (hashed colors)
        tmp_c2 = df[["lon","lat"]].copy()
        tmp_c2["pred_c2"] = pred_c2
        tmp_c2.to_parquet("_tmp_pred_c2.parquet", index=False)
        plot_geopandas("_tmp_pred_c2.parquet",
                       lon="lon", lat="lat",
                       color_by="pred_c2",
                       color_mode="hashed",
                       sample=None, markersize=3, alpha=0.9, figsize=(11,5))

        return {
            "pred_dist": pred_dist,
            "pred_c1": pred_c1,
            "pred_c2": pred_c2
        }

    # === COMPARISON MODE (default) ===
    err_km = np.abs(pred_dist - y_dist_km)
    c1_ok  = (pred_c1 == y_c1_id)
    c2_ok  = (pred_c2 == y_c2_id)

    # (A) distance error heatmap
    tmp_err = df[["lon","lat"]].copy()
    tmp_err["err_km"] = err_km
    tmp_err.to_parquet("_tmp_vis_err.parquet", index=False)
    plot_geopandas("_tmp_vis_err.parquet",
                   lon="lon", lat="lat",
                   color_by="err_km", color_mode="continuous",
                   log_scale=True, clip_quantiles=(0.01,0.99),
                   sample=None, markersize=2, alpha=0.9, figsize=(11,5))

    # (B) c1 correctness (green/red)
    _plot_green_red(df["lon"].to_numpy(), df["lat"].to_numpy(),
                    c1_ok, title="c1 correctness (green=correct, red=wrong)")

    # (C) c2 correctness (green/red)
    _plot_green_red(df["lon"].to_numpy(), df["lat"].to_numpy(),
                    c2_ok, title="c2 correctness (green=correct, red=wrong)")

    print(f"[distance]  MAE={err_km.mean():.3f} km | median={np.median(err_km):.3f} km | 95p={np.quantile(err_km,0.95):.3f} km")
    print(f"[c1]       acc={c1_ok.mean():.4f}")
    print(f"[c2]       acc={c2_ok.mean():.4f}")

    return {
        "err_km": err_km,
        "c1_ok": c1_ok,
        "c2_ok": c2_ok,
        "pred_c1": pred_c1,
        "pred_c2": pred_c2,
        "pred_dist": pred_dist
    }

    
    
def build_model_for_eval():
    depth = 5
    layer_counts = (256,)*depth
    return MultiHeadNIR(
        SIRENLayer,
        in_dim=3,
        layer_counts=layer_counts,
        params=((30.0,),)+((1.0,),)*(depth-1),
        code_bits=32
    )

compare_parquet_and_model_ecoc(
    parquet_path="python/geodata/parquet/dataset_all.parquet",
    checkpoint_path="python/nn_checkpoints/siren_best.pt",
    codes_path="python/geodata/countries.ecoc.json",
    model_builder=build_model_for_eval,
    sample=1_000_000,
    model_outputs_km=True
    #,predictions_only=True
)
