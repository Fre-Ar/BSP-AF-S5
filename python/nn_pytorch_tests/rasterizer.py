# python/nn_pytorch_tests/nir_raster_vizlike.py

from __future__ import annotations
from typing import Callable, Optional, Tuple, Literal, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import math

from nir import MultiHeadNIR, ClassHeadConfig  # your model types
from data import load_ecoc_codes               # JSON: {class_id: [0/1,...]}
from compare_data_viz import build_model_for_eval, model_path
from visualizer import overrides as new_hash
from world_bank_country_colors import colors_important

# ------------------------ utils ------------------------

def lonlat_to_xyz(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    lon = np.deg2rad(lon_deg); lat = np.deg2rad(lat_deg)
    cl = np.cos(lat)
    x = cl * np.cos(lon); y = cl * np.sin(lon); z = np.sin(lat)
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def bbox_height(width, lat_min, lat_max, lon_min, lon_max, mode="equal_angle"):
    # unwrap Δλ in [0, 360)
    dlon = (lon_max - lon_min) % 360.0
    dlat = lat_max - lat_min
    if mode == "equal_angle":
        return int(round(width * dlat / dlon))
    elif mode == "equal_distance":
        phi_mid = math.radians(0.5 * (lat_min + lat_max))
        return int(round(width * dlat / (dlon * math.cos(phi_mid))))
    else:
        raise ValueError("mode must be 'equal_angle' or 'equal_distance'")

def make_lonlat_grid_old(
    lon_min: float, lon_max: float, lat_min: float, lat_max: float, width: int, height: int
) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.linspace(lon_min, lon_max, num=width,  endpoint=True, dtype=np.float32)
    lat = np.linspace(lat_min, lat_max, num=height, endpoint=True, dtype=np.float32)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return lon2d, lat2d

_MERCATOR_LIMIT = 85.05112878  # Web Mercator latitude cap, degrees

def make_lonlat_grid(
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    width: int,
    height: int,
    lat_sampling: str = "linear",        # 'linear' | 'mercator'
    clamp_mercator: bool = True
):
    """
    Build lon/lat grid.
    - 'linear'   : lat is linearly spaced in degrees (equirectangular).
    - 'mercator' : lat is spaced uniformly in Mercator y, so the image drapes
                   with minimal vertical distortion over Web Mercator tiles.

    Returns lon2d, lat2d of shape (H, W).
    """
    if lat_sampling not in ("linear", "mercator"):
        raise ValueError("lat_sampling must be 'linear' or 'mercator'")

    if lat_sampling == "mercator":
        # Clamp to Web Mercator’s valid latitude
        if clamp_mercator:
            lat_min = float(np.clip(lat_min, -_MERCATOR_LIMIT, _MERCATOR_LIMIT))
            lat_max = float(np.clip(lat_max, -_MERCATOR_LIMIT, _MERCATOR_LIMIT))

        # helper: deg -> Mercator y, and inverse
        def _phi2y(phi_deg):
            phi = np.deg2rad(phi_deg)
            return np.arcsinh(np.tan(phi))  # asinh(tan φ)

        def _y2phi_deg(y):
            return np.rad2deg(np.arctan(np.sinh(y)))

        y_min = _phi2y(lat_min)
        y_max = _phi2y(lat_max)
        y = np.linspace(y_min, y_max, num=height, endpoint=True, dtype=np.float64)
        lat = _y2phi_deg(y).astype(np.float32)
    else:
        lat = np.linspace(lat_min, lat_max, num=height, endpoint=True, dtype=np.float32)

    lon = np.linspace(lon_min, lon_max, num=width, endpoint=True, dtype=np.float32)

    # IMPORTANT: arrange so row 0 corresponds to the *southern* edge (lat_min).
    # We’ll tell folium that the image origin is 'lower' so north is up.
    lon2d, lat2d = np.meshgrid(lon, lat)  # (H, W)
    return lon2d, lat2d

def codebook_to_bits_matrix(codebook: Dict[int, np.ndarray], n_bits: int | None = None):
    """
    From {class_id: np.uint8[K]} build:
      ids:  [C] int64 array of class ids (sorted ascending)
      bits: [C,K] float tensor (0/1), where K = n_bits or inferred from entries.
    """
    ids = np.array(sorted(codebook.keys()), dtype=np.int64)
    K_inf = int(next(iter(codebook.values())).shape[0])
    K = int(n_bits if n_bits is not None else K_inf)
    M = np.zeros((len(ids), K), dtype=np.uint8)
    for i, cid in enumerate(ids):
        v = codebook[cid]
        if v.shape[0] < K:
            raise ValueError(f"Code for class {cid} has length {v.shape[0]} < requested {K}")
        M[i, :] = v[:K].astype(np.uint8)
    bits = torch.from_numpy(M.astype(np.float32))
    return ids, bits  # np.int64[C], torch.float32[C,K]

@torch.no_grad()
def ecoc_decode_soft(
    logits_bits: torch.Tensor,   # [B, K] pre-sigmoid
    bits_codebook: torch.Tensor, # [C, K] in {0,1}
    tau: float = 1.0
) -> torch.Tensor:
    """
    Soft ECOC log-likelihood (no pos_weight shifts):
      score(c) = sum_k [ b_ck * log σ(z_k/tau) + (1-b_ck) * log σ(-z_k/tau) ]
    Returns indices in [0..C-1].
    """
    z = logits_bits / max(1e-6, float(tau))     # [B,K]
    B = bits_codebook.float()                   # [C,K]
    Z = z.unsqueeze(1)                          # [B,1,K]
    score = B.unsqueeze(0) * F.logsigmoid(Z) + (1 - B).unsqueeze(0) * F.logsigmoid(-Z)  # [B,C,K]
    return score.sum(dim=-1).argmax(dim=1)      # [B]

def _default_hash_rgba(ids: np.ndarray, opacity: float = 0.95) -> np.ndarray:
    """
    Deterministic int->RGBA hashing. Returns uint8 (N,4).
    """
    rs = np.random.RandomState(1337)
    uniq = np.unique(ids)
    lut: Dict[int, np.ndarray] = {}
    for k in uniq:
        rs.seed(int(k) * 2654435761 & 0xFFFFFFFF)
        rgb = rs.randint(20, 235, size=3).astype(np.uint8)
        lut[int(k)] = rgb
    rgba = np.zeros((ids.shape[0], 4), dtype=np.uint8)
    for i, k in enumerate(ids):
        rgb = lut[int(k)]
        rgba[i, :3] = rgb
        rgba[i, 3] = int(255 * opacity)
    return rgba

def _colormap_gray(values: np.ndarray, vmin: float, vmax: float, opacity: float = 0.95) -> np.ndarray:
    vals = np.clip((values - vmin) / max(1e-12, (vmax - vmin)), 0.0, 1.0)
    rgb = (vals[:, None] * 255.0).astype(np.uint8).repeat(3, axis=1)
    a = np.full((values.shape[0], 1), int(255 * opacity), dtype=np.uint8)
    return np.concatenate([rgb, a], axis=1)

# --- add these helpers near the top ---

def _hex_to_rgb8(hex_str: str) -> np.ndarray:
    """'#RRGGBB' or '#RGB' -> np.uint8[3]."""
    s = hex_str.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:  # #RGB -> #RRGGBB
        s = "".join(ch*2 for ch in s)
    if len(s) != 6:
        raise ValueError(f"Invalid hex color: {hex_str}")
    r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
    return np.array([r, g, b], dtype=np.uint8)

def _colorize_ids_with_overrides(
    ids: np.ndarray,
    opacity: float,
    overrides: dict[int, str] | None
) -> np.ndarray:
    """
    Base: hashed colors. Then apply per-ID hex overrides.
    Returns uint8 RGBA (N,4).
    """
    rgba = _default_hash_rgba(ids, opacity=opacity)  # (N,4)
    if overrides:
        ids64 = ids.astype(np.int64, copy=False)
        for k, hexcol in overrides.items():
            mask = (ids64 == int(k))
            if np.any(mask):
                rgb = _hex_to_rgb8(hexcol)
                rgba[mask, 0:3] = rgb  # keep same alpha
    return rgba

# ------------------------ main API ------------------------

def rasterize_model_from_checkpoint(
    checkpoint_path: str,
    model_builder,                         # callable that rebuilds SAME arch as training
    label_mode: str = "ecoc",              # 'auto' | 'ecoc' | 'softmax'
    codes_path: Optional[str] = None,      # required for ECOC
    # bbox & resolution
    lon_min: float = -180, lon_max: float = 180, lat_min: float = -85, lat_max: float = 85,
    width: int = 1920, height: int = 960,
    # what to render
    render: Literal["c1", "c2", "distance"] = "c1",
    tau: float = 1.0,                      # soft ECOC temperature
    model_outputs_log1p: bool = True,      # if True, distance head outputs log1p(km)
    distance_clip_km: Tuple[float, float] = (0.0, 500.0),
    # execution
    device: Optional[str] = None,
    batch_size: int = 65536,
    # colors
    overrides: dict = None,  # ids -> hex string
    opacity: float = 0.95,
) -> Tuple[np.ndarray, dict]:
    """
    Run the model over a lon/lat grid and return an RGBA image (H,W,4) + aux metadata.

    - ECOC mode: soft, non-pos_weighted decoder (matches your 'soft' visualizer).
      'codes_path' must point to the ECOC JSON codebook.
    - Softmax mode: argmax over logits; you may pass 'overrides' to color indices->RGBA.
    - 'overrides' signature: np.ndarray[int ids] -> np.ndarray[uint8 RGBA], same length.
    """
    # device
    if device is None:
        if torch.cuda.is_available(): device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): device = "mps"
        else: device = "cpu"

    # load checkpoint & config
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    lm = label_mode
    if lm == "auto":
        lm = cfg.get("label_mode", "ecoc")
    if lm not in {"ecoc", "softmax"}:
        raise ValueError(f"label_mode must be 'auto'|'ecoc'|'softmax', got {lm}")

    # build model and load weights
    model = model_builder().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    # ECOC bits (if needed)
    ids_np: Optional[np.ndarray] = None
    bits_t: Optional[torch.Tensor] = None
    if lm == "ecoc":
        if codes_path is None:
            raise ValueError("ECOC mode requires codes_path to the ECOC JSON codebook.")
        codebook = load_ecoc_codes(codes_path)  # {id: [0/1,...]}
        n_bits = int(cfg.get("n_bits", 32))
        ids_np, bits_t = codebook_to_bits_matrix(codebook, n_bits=n_bits)  # np.int64[C], torch.float32[C,K]
        bits_t = bits_t.to(device)

    # grid
    lon2d, lat2d = make_lonlat_grid(lon_min, lon_max, lat_min, lat_max, width, height)
    xyz = lonlat_to_xyz(lon2d, lat2d).reshape(-1, 3)  # (N,3)
    N = xyz.shape[0]
    out_rgba = np.zeros((N, 4), dtype=np.uint8)

    with torch.no_grad():
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            u = torch.from_numpy(xyz[s:e]).to(device)  # (B,3)

            # Multi-head NIR forward: distance head + two class heads
            pred_log1p, logits_c1, logits_c2 = model(u)  # (B,1),(B,K or C1),(B,K or C2)
            pred_log1p = pred_log1p.squeeze(-1)

            if render == "distance":
                d_km = torch.expm1(pred_log1p) if model_outputs_log1p else pred_log1p
                vals = d_km.clamp(min=0).float().cpu().numpy()
                rgba = _colormap_gray(vals, vmin=distance_clip_km[0], vmax=distance_clip_km[1], opacity=opacity)
                out_rgba[s:e] = rgba
                continue

            if lm == "ecoc":
                if render == "c1":
                    idxs = ecoc_decode_soft(logits_c1, bits_t, tau=tau).cpu().numpy()
                else:
                    idxs = ecoc_decode_soft(logits_c2, bits_t, tau=tau).cpu().numpy()
                class_ids = ids_np[idxs] if ids_np is not None else idxs
            else:
                # softmax heads: logits_* are (B, C)
                if render == "c1":
                    idxs = logits_c1.argmax(dim=1).cpu().numpy()
                else:
                    idxs = logits_c2.argmax(dim=1).cpu().numpy()
                class_ids = idxs  # you can map indices -> real IDs via an override if needed
                
            rgba = _colorize_ids_with_overrides(class_ids.astype(np.int64), opacity=opacity, overrides=overrides)   
            out_rgba[s:e] = rgba

    img = out_rgba.reshape(height, width, 4)
    aux = dict(
        lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
        width=width, height=height, render=render, label_mode=lm, tau=tau
    )
    return img, aux

# ------------------------ helpers ------------------------

def save_png(img_rgba: np.ndarray, path: str):
    Image.fromarray(img_rgba[::-1].copy()).save(path)

try:
    import folium
    _HAS_FOLIUM = True
except Exception:
    _HAS_FOLIUM = False

# TODO: Ignore this, it don't work, it always stretches map.
def to_folium_map(img_rgba: np.ndarray, aux: dict, tiles: str = "CartoDB positron", opacity: float = 1.0):
    import folium
    assert img_rgba.dtype == np.uint8 and img_rgba.ndim == 3 and img_rgba.shape[2] == 4

    bounds = [[aux["lat_min"], aux["lon_min"]], [aux["lat_max"], aux["lon_max"]]]

    lat_c = 0.5 * (aux["lat_min"] + aux["lat_max"])
    lon_c = 0.5 * (aux["lon_min"] + aux["lon_max"])
    m = folium.Map(location=[lat_c, lon_c], zoom_start=2, tiles=tiles, control_scale=True)

    folium.raster_layers.ImageOverlay(
        image=img_rgba,      # (H,W,4) uint8
        bounds=bounds,
        opacity=opacity,
        origin="lower",      # <-- key change to fix upside-down
        cross_origin=False,
        zindex=4,
        name=f"NIR {aux.get('render','')}",
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


# bbox (Luxembourg-ish)
lon_min, lon_max = 5.5, 6.7
lat_min, lat_max = 49.3, 50.3

# bbox Alpes
lat_max, lon_min = 50.368711, 4.872894
lat_min, lon_max = 46.369837, 16.733645

# bbox New Zealand
lat_max, lon_min = -32.069493, 165.165747
lat_min, lon_max = -53.153157, 178.802075

width = 1920
height = bbox_height(width, lat_min, lat_max, lon_min, lon_max)

render = "c1"
img, aux = rasterize_model_from_checkpoint(
    checkpoint_path=model_path,
    model_builder=build_model_for_eval,
    label_mode="ecoc",
    codes_path="python/geodata/countries.ecoc.json",
    render=render,
    tau=1.0,
    width=width, height=height,
    overrides=colors_important,  
    opacity=1.0
    ,lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
)
path = f"raster/siren_{render}_map_nz"
save_png(img, f"{path}.png")
print(f"Saved to {path}.png")

#m = to_folium_map(img, aux, tiles="CartoDB positron", opacity=1.0)
#m.save(f"{path}.html")
#print(f"Saved to {path}.html")