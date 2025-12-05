# src/nirs/viz/rasterizer.py

from __future__ import annotations
from typing import Optional, Tuple, Literal, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import math


from geodata.ecoc.ecoc import load_ecoc_codes, ecoc_decode, _ecoc_decode_soft_old, _codebook_to_bits_matrix_local

from nirs.world_bank_country_colors import colors_important
from nirs.create_nirs import build_model
from nirs.viz.visualizer import _hash_colors  
from nirs.engine import load_model_and_codebook

from utils.utils_geo import (
    COUNTRIES_ECOC_PATH,
    CHECKPOINT_PATH,
    MERCATOR_LIMIT,
    lonlat_to_unitvec,
    unitvec_to_lonlat)


# ------------------------ utils ------------------------

def bbox_height(width, lat_min, lat_max, lon_min, lon_max, mode="equal_angle"):
    """
    Given a lon/lat bounding box and a target image width, computes the height
    so that pixels have roughly comparable aspect ratio.

    Parameters
    ----------
    width : int
        Target image width (number of columns).
    lat_min, lat_max : float
        Bounds in degrees latitude.
    lon_min, lon_max : float
        Bounds in degrees longitude.
    mode : {"equal_angle", "equal_distance"}
        - "equal_angle"   : height/width = Δφ / Δλ  (simple equirectangular).
        - "equal_distance": compensates by cos(φ_mid) so that pixels are closer
                            to “square” in physical distance near the bbox center.

    Returns
    -------
    height : int
        Suggested image height in pixels.
    """
    # unwrap Δλ in [0, 360)
    dlon = (lon_max - lon_min) % 360.0
    dlat = lat_max - lat_min
    if dlon == 0:
        raise ValueError("bbox_height: zero longitudinal extent (lon_min == lon_max).")
    
    if mode == "equal_angle":
        return int(round(width * dlat / dlon))
    
    elif mode == "equal_distance":
        phi_mid = math.radians(0.5 * (lat_min + lat_max))
        # scale by cos φ_mid to approximate geodesic distances
        return int(round(width * dlat / (dlon * math.cos(phi_mid))))
    
    raise ValueError("mode must be 'equal_angle' or 'equal_distance'")

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
    Builds a regular lon/lat grid for rasterization.

    Parameters
    ----------
    lon_min, lon_max : float
        Longitudinal bounds in degrees.
    lat_min, lat_max : float
        Latitudinal bounds in degrees.
    width, height : int
        Grid resolution (W, H).
    lat_sampling : {"linear", "mercator"}
        - "linear"   : linearly spaced lat in degrees (equirectangular).
        - "mercator" : uniform spacing in Web-Mercator y; visually matches
                       standard web tiles better.
    clamp_mercator : bool
        If True, clamp lat_min/lat_max to the Web-Mercator valid range
        [-85.0511, 85.0511] when lat_sampling="mercator".

    Returns
    -------
    lon2d, lat2d : ndarray
        2D arrays of shape (H, W) with lon/lat in degrees.
    """
    if lat_sampling not in ("linear", "mercator"):
        raise ValueError("lat_sampling must be 'linear' or 'mercator'")

    if lat_sampling == "mercator":
        # Clamp to Web Mercator’s valid latitude
        if clamp_mercator:
            lat_min = float(np.clip(lat_min, -MERCATOR_LIMIT, MERCATOR_LIMIT))
            lat_max = float(np.clip(lat_max, -MERCATOR_LIMIT, MERCATOR_LIMIT))

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
        # simple equirectangular spacing
        lat = np.linspace(lat_min, lat_max, num=height, endpoint=True, dtype=np.float32)

    lon = np.linspace(lon_min, lon_max, num=width, endpoint=True, dtype=np.float32)

    # Convention: row 0 is the *southern* edge (lat_min).
    lon2d, lat2d = np.meshgrid(lon, lat)  # (H, W)
    return lon2d, lat2d

def _colormap_gray(
    values: np.ndarray,
    vmin: float,
    vmax: float,
    opacity: float = 0.95,
) -> np.ndarray:
    """
    Simple grayscale mapping for scalar values.

    Parameters
    ----------
    values : ndarray
        1D array of scalar values to map into [vmin, vmax].
    vmin, vmax : float
        Lower/upper clipping bounds.
    opacity : float
        Alpha channel in [0,1].

    Returns
    -------
    rgba : ndarray
        (N, 4) uint8 array of RGBA pixels.
    """
    vals = np.clip((values - vmin) / max(1e-12, (vmax - vmin)), 0.0, 1.0)
    
    rgb = (vals[:, None] * 255.0).astype(np.uint8).repeat(3, axis=1)
    a = np.full((values.shape[0], 1), int(255 * opacity), dtype=np.uint8)
    
    return np.concatenate([rgb, a], axis=1)

def _colorize_ids_with_overrides(
    ids: np.ndarray,
    opacity: float,
    overrides: Dict[int, str] | None,
) -> np.ndarray:
    """
    Colorizes integer class IDs to uint8 RGBA using the global hashed palette from
    `visualizer._hash_colors`, plus optional per-ID overrides.

    Parameters
    ----------
    ids : np.ndarray
        1D array of integer IDs (e.g., country codes).
    opacity : float
        Alpha in [0,1].
    overrides : dict[int, str] | None
        Optional mapping {id -> matplotlib color} (e.g. "#RRGGBB").

    Returns
    -------
    rgba : np.ndarray
        (N, 4) uint8 array of colors in [0,255].
    """
    # _hash_colors returns float RGBA in [0,1]
    rgba_f = _hash_colors(ids, overrides=overrides)  # (N, 4) float
    rgba = np.clip(np.round(rgba_f * 255.0), 0, 255).astype(np.uint8)

    if opacity < 1.0:
        rgba[:, 3] = int(255 * opacity)

    return rgba

# ------------------------ main API ------------------------

def rasterize_model_from_checkpoint(
    # model params
    checkpoint_path: str,
    model_name: str,
    layer_counts,
    depth: int, layer: int,
    w0: float, w_h: float, s_param: float, beta: float,
    global_z: bool,
    regularize_hyperparams: bool,
    
    # labels / decoding
    label_mode: str = "ecoc",             # "auto" | "ecoc" | "softmax"
    codes_path: Optional[str] = None,     # required for ECOC
    
    # bbox & resolution
    lon_min: float = -180, lon_max: float = 180,
    lat_min: float = -85, lat_max: float = 85,
    width: int = 1920, height: int = 960,
    
    # what to render
    render: Literal["c1", "c2", "distance"] = "c1",
    tau: float = 1.0,                     # temperature for ECOC logits
    model_outputs_log1p: bool = True,     # if True, distance head outputs log1p(km)
    distance_clip_km: Tuple[float, float] = (0.0, 500.0),
    
    # execution
    device: Optional[str] = None,
    batch_size: int = 65_536,
    
    # colors
    overrides: Dict[int, str] | None = None,  # ids -> hex / mpl color
    opacity: float = 0.95,
) -> Tuple[np.ndarray, dict]:
    """
    Runs a trained NIR checkpoint over a lon/lat grid and rasterize either:
      
      - distance to nearest border ("distance"), or
      - predicted c1 / c2 region IDs ("c1" / "c2").

    Parameters
    ----------
    checkpoint_path : str
        Path to the saved `.pt` checkpoint containing "model" and "config".
    model_name : str
        Key understood by `build_model` (e.g., "siren", "split_siren", "incode").
    layer_counts, depth, layer, w0, w_h, s_param, beta, global_z, regularize_hyperparams
        Hyperparameters passed through to `build_model`.
    label_mode : {"ecoc", "softmax"}
        - "ecoc": use ECOC decoding with `codes_path`.
        - "softmax": treat class heads as softmax over class indices.
    codes_path : str, optional
        Path to ECOC JSON codebook (required for ECOC mode).
    lon_min, lon_max, lat_min, lat_max : float
        Bounding box in degrees.
    width, height : int
        Output raster size.
    render : {"c1", "c2", "distance"}
        Which head to visualize.
    tau : float
        Temperature scaling for ECOC logits: logits / tau before decoding.
    model_outputs_log1p : bool
        If True, distance head output is log1p(km); otherwise assume km directly.
    distance_clip_km : (float, float)
        Range [vmin, vmax] for distance visualization (grayscale).
    device : str, optional
        "cuda", "mps", "cpu", or None for auto-detection.
    batch_size : int
        Number of grid points evaluated per forward pass.
    overrides : dict[int, str], optional
        Optional mapping from IDs to explicit colors (e.g. world_bank palette).
    opacity : float
        Alpha in [0,1] for the output image.

    Returns
    -------
    img : np.ndarray
        RGBA image of shape (H, W, 4), dtype=uint8, bottom-up (row 0 = lat_min).
    aux : dict
        Metadata: bbox, resolution, effective label_mode, tau, etc.
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
    
    cfg = ckpt.get("config", {})
    pw_c1, pw_c2 = cfg.get("pos_weight_c1", None), cfg.get("pos_weight_c2", None)
    
    # ---- ECOC bits (old-style) ----
    ids_np: Optional[np.ndarray] = None
    bits_t: Optional[torch.Tensor] = None
    if label_mode == "ecoc":
        n_bits = int(cfg.get("n_bits", 32))
        ids_np, bits_t = _codebook_to_bits_matrix_local(codebook, n_bits=n_bits)
        bits_t = bits_t.to(device)
    
    # build lon/lat grid and unit vectors
    lon2d, lat2d = make_lonlat_grid(
        lon_min=lon_min, lon_max=lon_max,
        lat_min=lat_min, lat_max=lat_max,
        width=width,     height=height,
        lat_sampling="linear"
    )
    xyz = lonlat_to_unitvec(lon2d, lat2d).reshape(-1, 3).astype(np.float32)  # (N,3)
    N = xyz.shape[0]

    out_rgba = np.zeros((N, 4), dtype=np.uint8)

    debug = False
    if debug:
        # --- debug: probe a single point from the grid ---
        debug_lon_val = lon_max
        debug_lat_val = lat_min

        # your lonlat_to_xyz / lonlat_to_unitvec only accepts np.ndarray
        debug_lon = np.array([debug_lon_val], dtype=np.float32)
        debug_lat = np.array([debug_lat_val], dtype=np.float32)

        debug_xyz = lonlat_to_unitvec(debug_lon, debug_lat).astype(np.float32)  # shape (1,3)
        print(debug_xyz)
        point_dbg = np.array([0.0232, 0.0099, 0.0228], dtype=np.float32)
        print("try", unitvec_to_lonlat(point_dbg))
        #debug_xyz = np.array([0.0232, 0.0099, 0.0228], dtype=np.float32)
        u_dbg = torch.from_numpy(debug_xyz).to(device)  # (1,3)
        
        with torch.no_grad():
            d_hat_dbg, logits_c1_dbg, logits_c2_dbg = model(u_dbg)
            d_hat_dbg = d_hat_dbg.squeeze(-1)

            if label_mode == "ecoc":
                # ECOC decode with pos_weight if available
                if render == "c1":
                    logits_dbg = logits_c1_dbg / max(1e-6, float(tau))
                    pw_dbg = torch.tensor(pw_c1, dtype=torch.float32, device=device) if pw_c1 is not None else None
                else:
                    logits_dbg = logits_c2_dbg / max(1e-6, float(tau))
                    pw_dbg = torch.tensor(pw_c2, dtype=torch.float32, device=device) if pw_c2 is not None else None
                
                
                class_id_dbg = ecoc_decode(
                    logits_dbg,
                    codebook=codebook,
                    pos_weight=pw_dbg,
                    mode="soft",
                )[0].item()
            else:
                # softmax case
                if render == "c1":
                    class_id_dbg = int(logits_c1_dbg.argmax(dim=1).item())
                else:
                    class_id_dbg = int(logits_c2_dbg.argmax(dim=1).item())

        print(f"DEBUG single-point: lon={debug_lon_val:.4f}, lat={debug_lat_val:.4f}, c1_id={class_id_dbg}")
        print("DEBUG logits_c1_dbg[:10]:", logits_c1_dbg[0, :10].detach().cpu().numpy())


    with torch.no_grad():
        for s_idx in range(0, N, batch_size):
            e_idx = min(N, s_idx + batch_size)
            u = torch.from_numpy(xyz[s_idx:e_idx]).to(device)  # (B,3)

            if s_idx == 0:
                norms = u.norm(dim=1)
                print("DEBUG raster norms: min", norms.min().item(),
                    "max", norms.max().item(),
                    "mean", norms.mean().item())
            # Multi-head forward: distance + two class heads
            
            pred_log1p, logits_c1, logits_c2 = model(u)
            pred_log1p = pred_log1p.squeeze(-1)
            
            # --- distance-only rendering ---
            if render == "distance":
                if model_outputs_log1p:
                    d_km = torch.expm1(pred_log1p)
                else:
                    d_km = pred_log1p
                    
                vals = d_km.clamp(min=0).float().cpu().numpy()
                rgba = _colormap_gray(
                    vals,
                    vmin=distance_clip_km[0],
                    vmax=distance_clip_km[1],
                    opacity=opacity,
                )
                out_rgba[s_idx:e_idx] = rgba
                continue

            # --- classification rendering (c1 / c2) ---
            if label_mode == "ecoc":
                # temperature scaling: logits / tau
                if render == "c1":
                    logits = logits_c1 / max(1e-6, float(tau))
                    pw = torch.tensor(pw_c1, dtype=torch.float32, device=device) if pw_c1 is not None else None
                else:
                    logits = logits_c2 / max(1e-6, float(tau))
                    pw = torch.tensor(pw_c2, dtype=torch.float32, device=device) if pw_c1 is not None else None

                
                t = '''class_ids_t = ecoc_decode(
                    logits,
                    codebook=codebook,
                    pos_weight=None,
                    mode="soft",
                )
                class_ids = class_ids_t.long().cpu().numpy()'''
                
                idxs = _ecoc_decode_soft_old(logits, bits_t, tau).cpu().numpy()
                class_ids = ids_np[idxs]
                if s_idx == 0:
                    unique, counts = np.unique(class_ids, return_counts=True)
                    print("DEBUG raster first batch unique IDs:", unique[:10], "counts:", counts[:10])
                    #print("logits_c1[0][:10] =", logits_c1[0, :10].detach().cpu().numpy())
                
            
            else:
                # softmax: logits_* are (B, C) over contiguous class indices
                if render == "c1":
                    class_ids = logits_c1.argmax(dim=1).long().cpu().numpy()
                else:
                    class_ids = logits_c2.argmax(dim=1).long().cpu().numpy()

            rgba = _colorize_ids_with_overrides(
                class_ids.astype(np.int64),
                opacity=opacity,
                overrides=overrides,
            )
            out_rgba[s_idx:e_idx] = rgba

    img = out_rgba.reshape(height, width, 4)
    aux = dict(
        lon_min=lon_min, lon_max=lon_max,
        lat_min=lat_min, lat_max=lat_max,
        width=width,     height=height,
        render=render,
        label_mode=label_mode,
        tau=tau,
    )
    return img, aux

# ------------------------ wrappers ------------------------

def save_png(img_rgba: np.ndarray, path: str):
    """
    Saves an RGBA image to disk, flipping vertically so that row 0 corresponds
    to the southern edge (lat_min) when viewed normally.

    Parameters
    ----------
    img_rgba : ndarray
        Image array of shape (H, W, 4), uint8.
    path : str
        Output PNG path.
    """
    if img_rgba.dtype != np.uint8 or img_rgba.ndim != 3 or img_rgba.shape[2] != 4:
        raise ValueError("save_png expects an (H, W, 4) uint8 RGBA image.")
    # Flip vertically: our grid origin is south, PNG viewers expect north at top.
    Image.fromarray(img_rgba[::-1].copy()).save(path)

def raster(model_name, mode,
        layer_counts,
        depth, layer,
        w0, w_h, s, beta,
        global_z,regularize_hyperparams,
        render: str = "c1",
        area: str = "alpes"):
    """
    Convenience wrapper: pick a preset bbox (Luxembourg, Alpes, NZ),
    rasterize a checkpoint, and save a PNG under `raster/`.

    Parameters
    ----------
    model_name, mode, layer_counts, depth, layer, w0, w_h, s, beta, global_z,
    regularize_hyperparams
        Hyperparameters for checkpoint naming and `build_model` construction.
    render : {"c1", "c2", "distance"}
        Head to visualize.
    area : {"lux", "alpes", "nz"}
        Predefined geographic region.
    """
    # --- predefined bboxes ---
    if area == "lux":
        lon_min, lon_max = 5.5, 6.7
        lat_min, lat_max = 49.3, 50.3

    elif area == "alpes":
        lat_max, lon_min = 50.368711, 4.872894
        lat_min, lon_max = 46.369837, 16.733645

    elif area == "nz":
        lat_max, lon_min = -32.069493, 165.165747
        lat_min, lon_max = -53.153157, 178.802075
    else:
        raise ValueError(f"Unknown area preset: {area!r}")

    width = 1920
    height = bbox_height(width, lat_min, lat_max, lon_min, lon_max)

    model_path = f"{CHECKPOINT_PATH}/{model_name}_{mode}_1M_{depth}x{layer}_w0{w0}_wh{w_h}.pt" 
    img, aux = rasterize_model_from_checkpoint(
        model_name=model_name,
        checkpoint_path=model_path,
        label_mode=mode,
        codes_path=COUNTRIES_ECOC_PATH,
        render=render,
        tau=1.0,
        width=width, height=height,
        overrides=colors_important,  
        opacity=1.0
        ,lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
        
        layer_counts = layer_counts,
        depth = depth, layer = layer,
        w0=w0, w_h=w_h, s_param=s, beta=beta,
        global_z=global_z,
        regularize_hyperparams=regularize_hyperparams
    )
    path = f"raster/{model_name}_{depth}x{layer}_{render}_map_{area}"
    save_png(img, f"{path}.png")
    print(f"Saved to {path}.png")