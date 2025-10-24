# python/nn_pytorch_tests/compare_data_viz.py


import json, math, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from visualizer import plot_geopandas, overrides as new_hash
from nn_siren import SIRENLayer, SIREN
from nn_relu import ReLULayer
from nir import NIRLayer, NIRTrunk, MultiHeadNIR, ClassHeadConfig
from data import BordersParquet, LossWeights, train_one_epoch, evaluate, load_ecoc_codes
import matplotlib.pyplot as plt
import geopandas as gpd
import torch.nn.functional as F
from stats import ecoc_prevalence_by_bit, pos_weight_from_prevalence

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

@torch.no_grad()
def _ecoc_decode_soft_logits(
    logits: torch.Tensor,      # [B, K] pre-sigmoid
    bits: torch.Tensor,        # [C, K] 0/1 codes (same device/dtype as logits)
    pos_weight=None            # None | scalar | [K] tensor
) -> torch.Tensor:
    """
    Soft ECOC decoding that matches BCEWithLogitsLoss(pos_weight):
      - Apply per-bit logit shift tau_k = -log(w_k) (if pos_weight given)
      - Score each class with sum_k [ b_k*logsigmoid(z_k - tau_k) + (1-b_k)*logsigmoid(-(z_k - tau_k)) ]
      - Return argmax over classes (0..C-1), i.e., indices into 'ids'
    """
    if pos_weight is None:
        z_adj = logits
    else:
        if not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(pos_weight, dtype=logits.dtype, device=logits.device)
        else:
            pos_weight = pos_weight.to(dtype=logits.dtype, device=logits.device)
        if pos_weight.numel() == 1:
            pos_weight = pos_weight.expand(logits.shape[1])   # [K]
        tau = (-torch.log(pos_weight)).view(1, -1)            # [1,K]
        z_adj = logits - tau

    Z = z_adj.unsqueeze(1)            # [B,1,K]
    BITS = bits.unsqueeze(0)          # [1,C,K]
    score = BITS * F.logsigmoid(Z) + (1.0 - BITS) * F.logsigmoid(-Z)  # [B,C,K]
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
    model_builder,                 # callable that rebuilds SAME arch as training
    label_mode: str = "auto",
    codes_path: str| None = None,
    sample: int | None = 200_000,
    batch_size: int = 131_072,
    device: str | None = None,
    n_bits: int | None = None,     # if None, inferred from codebook
    model_outputs_log1p: bool = True,
    earth_radius_km: float = 6371.0,
    predictions_only: bool = False, # <-- just show model predictions
    overrides=None,
    use_bayes_thr: bool = True
):
    # device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else: device = "cpu"

    # load parquet
    cols = ["lon","lat","x","y","z","dist_km", "log1p_dist","c1_id","c2_id"]
    df = pd.read_parquet(parquet_path, columns=cols)
    if sample and len(df) > sample:
        df = df.sample(sample, random_state=0).reset_index(drop=True)

    xyz = df[["x","y","z"]].to_numpy(dtype=np.float32, copy=True)
    nrm = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = (xyz / np.clip(nrm, 1e-9, None)).astype(np.float32)

    y_dist_km = df["dist_km"].to_numpy(dtype=np.float32)
    y_log1p_dist = df["log1p_dist"].to_numpy(dtype=np.float32)
    y_c1_id   = df["c1_id"].to_numpy(dtype=np.int64)
    y_c2_id   = df["c2_id"].to_numpy(dtype=np.int64)

    # checkpoint and config
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    lm = label_mode
    if lm == "auto":
        lm = cfg.get("label_mode", "ecoc")
    if lm not in {"ecoc", "softmax"}:
        raise ValueError(f"label_mode must be 'auto'|'ecoc'|'softmax', got {lm}")

    if lm == "ecoc":
        n_bits = int(cfg.get("n_bits", 32))
        class_cfg = ClassHeadConfig(class_mode="ecoc", n_bits=n_bits)
    else:
        n_c1 = int(cfg.get("n_classes_c1", 289))
        n_c2 = int(cfg.get("n_classes_c2", 289))
        class_cfg = ClassHeadConfig(class_mode="softmax", n_classes_c1=n_c1, n_classes_c2=n_c2)

    # model
    model = model_builder().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # For ECOC mode, build bits matrix
    if lm == "ecoc":
        if codes_path is None:
            raise ValueError("ECOC mode requires codes_path to the ECOC JSON codebook.")
        codebook = load_ecoc_codes(codes_path)
        ids, bits = codebook_to_bits_matrix(codebook, n_bits=class_cfg.n_bits)
        ids = torch.from_numpy(ids).to(device)
        bits = bits.to(device)
    else:
        ids = None
        bits = None

    # inference
    N = xyz.shape[0]
    pred_dist = np.zeros(N, dtype=np.float32)
    pred_c1   = np.zeros(N, dtype=np.int64)
    pred_c2   = np.zeros(N, dtype=np.int64)

    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            u = torch.from_numpy(xyz[s:e]).to(device)
            
            d_hat, logits_c1, logits_c2 = model(u)
            d_hat = d_hat.squeeze(-1)

            if model_outputs_log1p:
                d_km = torch.expm1(d_hat)  # convert log1p(dist_km) -> dist_km
            else:
                # fallback: assume it's already in km (legacy)
                d_km = d_hat

            if lm == "ecoc":
                # decode via soft ECOC score
                if use_bayes_thr:
                    ones_c1, totals_c1, p1_c1 = ecoc_prevalence_by_bit(parquet_path, codebook, id_col="c1_id")
                    ones_c2, totals_c2, p1_c2 = ecoc_prevalence_by_bit(parquet_path, codebook, id_col="c2_id")

                    pw_c1 = pos_weight_from_prevalence(p1_c1)
                    pw_c2 = pos_weight_from_prevalence(p1_c2)
                else:
                    pw_c1 = None
                    pw_c2 = None
                c1_idx = _ecoc_decode_soft_logits(logits_c1, bits, pos_weight=pw_c1)  # indices into 'ids'
                c2_idx = _ecoc_decode_soft_logits(logits_c2, bits, pos_weight=pw_c2)
                pred_c1[s:e] = ids[c1_idx].long().cpu().numpy()
                pred_c2[s:e] = ids[c2_idx].long().cpu().numpy()
            else:
                # softmax argmax (class indices assumed to match dataset ids 0..C-1)
                pred_c1[s:e] = logits_c1.argmax(dim=1).long().cpu().numpy()
                pred_c2[s:e] = logits_c2.argmax(dim=1).long().cpu().numpy()

            pred_dist[s:e] = d_km.float().cpu().numpy()

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
                       sample=None, markersize=2, alpha=0.9, figsize=(11,5), overrides=overrides)

        # (B) predicted c1 (hashed colors)
        tmp_c1 = df[["lon","lat"]].copy()
        tmp_c1["pred_c1"] = pred_c1
        tmp_c1.to_parquet("_tmp_pred_c1.parquet", index=False)
        plot_geopandas("_tmp_pred_c1.parquet",
                       lon="lon", lat="lat",
                       color_by="pred_c1",
                       color_mode="hashed",
                       sample=None, markersize=3, alpha=0.9, figsize=(11,5), overrides=overrides)

        # (C) predicted c2 (hashed colors)
        tmp_c2 = df[["lon","lat"]].copy()
        tmp_c2["pred_c2"] = pred_c2
        tmp_c2.to_parquet("_tmp_pred_c2.parquet", index=False)
        plot_geopandas("_tmp_pred_c2.parquet",
                       lon="lon", lat="lat",
                       color_by="pred_c2",
                       color_mode="hashed",
                       sample=None, markersize=3, alpha=0.9, figsize=(11,5), overrides=overrides)

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
                   sample=None, markersize=2, alpha=0.9, figsize=(11,5), overrides=overrides)

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

mode = "ecoc" 
    
DEPTH = 15
LAYER = 128
w0 = 60.0  
    
def build_model_for_eval():
    layer_counts = (LAYER,)*DEPTH
    cfg = ClassHeadConfig(class_mode=mode,
                        n_bits=32,
                        n_classes_c1=289,
                        n_classes_c2=289)
    w_h = 1.0
    model = MultiHeadNIR(
        SIRENLayer,
        in_dim=3,
        layer_counts=layer_counts,
        params=((w0,),)+((w_h,),)*(DEPTH-1),
        class_cfg = cfg
        #,head_layers=(LAYER,)
        )
    '''model = MultiHeadNIR(
        ReLULayer,
        in_dim=3,
        layer_counts=layer_counts,
        params=((),)+((),)*(depth-1),
        class_cfg = cfg)'''
    return model

#model_path = f"python/nn_checkpoints/siren_{mode}_1M_{DEPTH}x{LAYER}_0h_w{w0}_bayes_thr_200e.pt"
model_path = f"python/nn_checkpoints/siren_{mode}_1M_{DEPTH}x{LAYER}_0h_w{w0}_post.pt"

def run():
    compare_parquet_and_model_ecoc(
        parquet_path="python/geodata/parquet/log_dataset_1M.parquet",
        checkpoint_path=model_path,
        codes_path="python/geodata/countries.ecoc.json",
        model_builder=build_model_for_eval,
        sample=1_000_000,
        model_outputs_log1p = True
        ,predictions_only=True
        ,overrides=new_hash
        , use_bayes_thr = False
    )
    
run()
