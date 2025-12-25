# src/nirs/viz/rasterizer.py

from __future__ import annotations
from typing import Optional, Tuple, Literal, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import math


from geodata.ecoc.ecoc import ecoc_decode, _ecoc_decode_soft_old, _codebook_to_bits_matrix_local, _prepare_codebook_tensor

from nirs.world_bank_country_colors import colors_important
from nirs.viz.visualizer import _hash_colors  
from nirs.engine import Predictor
from nirs.inference import InferenceConfig

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
    model_cfg: InferenceConfig,
    
    # bbox & resolution
    lon_min: float = -180, lon_max: float = 180,
    lat_min: float = -85, lat_max: float = 85,
    width: int = 1920, height: int = 960,
    
    # what to render
    render: Literal["c1", "c2", "distance"] = "c1",
    tau: float = 1.0,                     # temperature for ECOC logits
    distance_clip_km: Tuple[float, float] = (0.0, 500.0),
    
    # execution
    device: Optional[str] = None,
    batch_size: int = 65_536,
    
    # colors
    overrides: Dict[int, str] | None = None,  # ids -> hex / mpl color
    opacity: float = 1.0,
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
    # 1. Load model from checkpoint
    predictor = Predictor(model_cfg, checkpoint_path, device)

    # 2. Build Grid
    lon2d, lat2d = make_lonlat_grid(
        lon_min, lon_max, lat_min, lat_max, width, height, "linear"
    )
    # Flatten to list of unit vectors
    xyz = lonlat_to_unitvec(lon2d, lat2d).reshape(-1, 3).astype(np.float32)
    N = xyz.shape[0]
    out_rgba = np.zeros((N, 4), dtype=np.uint8)

    # 3. Batched Inference Loop
    print(f"[Rasterizer] Rendering {N} pixels in batches of {batch_size}...")
    for s_idx in range(0, N, batch_size):
        e_idx = min(N, s_idx + batch_size)
        batch_xyz = xyz[s_idx:e_idx]
        
        # Predict
        pred = predictor.predict(batch_xyz, tau=tau)
        
        # Colorize based on mode
        if render == "distance":
            rgba = _colormap_gray(
                pred.dist_km, 
                vmin=distance_clip_km[0], 
                vmax=distance_clip_km[1], 
                opacity=opacity
            )
        elif render == "c1":
            rgba = _colorize_ids_with_overrides(pred.c1_ids, opacity, overrides)
        elif render == "c2":
            rgba = _colorize_ids_with_overrides(pred.c2_ids, opacity, overrides)
        else:
            raise ValueError(f"Unknown render mode: {render}")
            
        out_rgba[s_idx:e_idx] = rgba
    
    # 4. Reshape
    img = out_rgba.reshape(height, width, 4)
    aux = {
        "lon_min": lon_min, "lon_max": lon_max,
        "lat_min": lat_min, "lat_max": lat_max,
        "width": width, "height": height,
        "render": render, "label_mode": model_cfg.label_mode, "tau": tau
    }
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

def raster(model_cfg: InferenceConfig,
        checkpoint_path: str,
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

    img, aux = rasterize_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_cfg=model_cfg,
        
        render=render,
        overrides=colors_important,
        
        width=width, height=height,
        lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max,
    )
    cfg_name = checkpoint_path.split('/')[-1].removesuffix('.pt')
    out_path = f"raster/{cfg_name}_{render}_map_{area}.png"
    save_png(img, out_path)
    print(f"Saved to {out_path}")