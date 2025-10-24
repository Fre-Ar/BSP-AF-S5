# python/geodata/sampler.py
"""
Minimal, production-ready generator for (lon, lat, x, y, z, dist_km, log1p_dist, c1_id, c2_id, is_border, r_band)
saved as **one Parquet file**.

Key choices:
- We bias sampling near borders (default 70/30).
- Distances are **spherical** to short-arc border segments.
- Labeling is parallel: each worker initializes once, labels a shard, writes a temp Parquet; parent concatenates.
- **Shortlist is KD-tree only** (KDTree on segment midpoints in 3D unit sphere). No STRtree anywhere.
- Optional Numba JIT for the exact distance kernel (falls back to NumPy if absent).
- Windows/macOS-safe (spawn start method).
"""

from __future__ import annotations
from utils import _save_points_parquet, _load_points_parquet  # your helper I/O

# --- stdlib
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# --- third-party
import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Point
from shapely.prepared import prep
from scipy.spatial import KDTree

try:
    from tqdm import tqdm  # optional
except Exception:
    tqdm = None

# Optional Numba acceleration
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

# --------------------------- configuration ---------------------------

R_EARTH_KM = 6371.0088                 # mean Earth radius
FOLDER_PATH = "python/geodata"         # base I/O folder (change if you like)

GPKG_PATH   = os.path.join(FOLDER_PATH, "world_bank_geodata.gpkg")
BORDERS_FGB = os.path.join(FOLDER_PATH, "borders.fgb")
COUNTRIES_LAYER = "countries"
ID_FIELD    = "id"                     # numeric country id in your GPKG

VALIDATE_NEAR_BORDER = True
HINT_DELTA_FACTOR = 0.8  # if computed_d < 0.8 * hint_d, override

# --------------------------- small utilities ------------------------

def _safe_norm(v: np.ndarray, axis=1, keepdims=True, eps=1e-15):
    n = np.linalg.norm(v, axis=axis, keepdims=keepdims)
    return np.where(n < eps, eps, n)

def _safe_div(v: np.ndarray, n: np.ndarray, eps=1e-15):
    n = np.where(n < eps, 1.0, n)
    return v / n

# --- spherical conversions

def lonlat_to_unitvec(lon_deg, lat_deg) -> np.ndarray:
    lon = np.radians(lon_deg); lat = np.radians(lat_deg)
    cl  = np.cos(lat)
    v = np.stack([cl*np.cos(lon), cl*np.sin(lon), np.sin(lat)], axis=-1)
    v = _safe_div(v, _safe_norm(v))
    return v.astype(np.float32)

def unitvec_to_lonlat(v: np.ndarray):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return lon.astype(np.float32), lat.astype(np.float32)

def move_along_geodesic(p: np.ndarray, t: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rotate unit vectors p along unit tangents t by angles theta (radians)."""
    ct = np.cos(theta)[:, None]; st = np.sin(theta)[:, None]
    out = p * ct + t * st
    out = _safe_div(out, _safe_norm(out))
    return out.astype(np.float32)

# Scalar reference for tests/debug
def greatcircle_point_segment_dist_km(p_lon, p_lat, a_lon, a_lat, b_lon, b_lat) -> float:
    def _ll2v(lon, lat):
        lon = np.radians(lon); lat = np.radians(lat)
        cl = np.cos(lat)
        return np.array([cl*np.cos(lon), cl*np.sin(lon), np.sin(lat)], dtype=np.float64)

    p = _ll2v(p_lon, p_lat)
    a = _ll2v(a_lon, a_lat)
    b = _ll2v(b_lon, b_lat)

    n = np.cross(a, b); nn = np.linalg.norm(n)
    if nn == 0.0:
        return R_EARTH_KM * min(
            np.arccos(np.clip(np.dot(p, a), -1.0, 1.0)),
            np.arccos(np.clip(np.dot(p, b), -1.0, 1.0)),
        )
    n /= nn

    c = np.cross(n, p); c = np.cross(c, n); c /= np.linalg.norm(c)

    ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    ac = np.arccos(np.clip(np.dot(a, c), -1.0, 1.0))
    cb = np.arccos(np.clip(np.dot(c, b), -1.0, 1.0))

    if abs((ac + cb) - ab) < 1e-10:
        theta = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
    else:
        theta = min(
            np.arccos(np.clip(np.dot(p, a), -1.0, 1.0)),
            np.arccos(np.clip(np.dot(p, b), -1.0, 1.0)),
        )
    return float(R_EARTH_KM * theta)

# --------------------------- sampling logic -------------------------

def _sample_distance_km_test(batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Piecewise log-uniform bands favoring near-border points.

    Bands (r = 0..6):
      r0:  [0.01, 1)   km
      r1:  [1,    5)   km
      r2:  [5,    10)  km
      r3:  [10,   25)  km
      r4:  [25,   50)  km
      r5:  [50,  150)  km 
      r6:  [150,  500] km

    Sampling probabilities (sum to 1 across these bands):
      p ∝ {22, 18, 15, 8, 4, 2, 1}  ->  [0.3142857, 0.2571429, 0.2142857, 0.1142857, 0.0571429, 0.0285714, 0.0142857]

    Note: r255 ("uniform globe") is handled elsewhere and not sampled here.
    """
    # Bands low/high (km). Use a small >0 lower bound for r0 to avoid log(0).
    lows  = np.array([0.01,  1.0,   5.0,   10.0,  25.0, 150.0, 150.0], dtype=np.float32)
    highs = np.array([1.0,   5.0,  10.0,   25.0,  50.0, 150.0, 500.0], dtype=np.float32)
    # Probabilities normalized from {22,18,15,8,4,2,1}
    p_raw = np.array([22, 18, 15, 8, 4, 2, 1], dtype=np.float64)
    probs = (p_raw / p_raw.sum()).astype(np.float64)

    bands = rng.choice(np.arange(7, dtype=np.uint8), size=batch_size, p=probs)
    low  = lows[bands]
    high = highs[bands]

    # Log-uniform sampling within each band
    u = rng.random(batch_size, dtype=np.float32)
    d = np.exp(np.log(low) + u * (np.log(high) - np.log(low))).astype(np.float32)
    return d, bands


def _sample_distance_km(batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Piecewise log-uniform bands favoring near-border points.

    Bands (r = 0..6):
      r0:  [0.01, 1)   km
      r1:  [1,    5)   km
      r2:  [5,    10)  km
      r3:  [10,   25)  km
      r4:  [25,   50)  km
      r5:  [50,  150)  km 
      r6:  [150,  500] km

    Sampling probabilities (sum to 1 across these bands):
      p ∝ {22, 18, 15, 8, 4, 2, 1}

    Note: r255 ("uniform globe") is handled elsewhere and not sampled here.
    """
    bands = rng.choice([0, 1, 2], size=batch_size, p=[0.6, 0.3, 0.1])
    low  = np.array([0.1, 10.0, 50.0], dtype=np.float32)[bands]
    high = np.array([10.0, 50.0, 300.0], dtype=np.float32)[bands]
    u = rng.random(batch_size, dtype=np.float32)
    d = np.exp(np.log(low) + u * (np.log(high) - np.log(low))).astype(np.float32)
    return d, bands.astype(np.uint8)

# ---------------------- vector distance kernel ----------------------

def _theta_point_to_short_arc_numpy(p: np.ndarray, A: np.ndarray, B: np.ndarray, N: np.ndarray, theta_ab: np.ndarray) -> np.ndarray:
    """
    NumPy vectorized: return array of angles (rad) from point p (3,) to short-arc A-B for K candidates.
    Inputs are float64; A,B,N shape (K,3); theta_ab shape (K,).
    """
    C = np.cross(N, p)
    C = np.cross(C, N)
    C = _safe_div(C, _safe_norm(C))

    def aco(x): return np.arccos(np.clip(x, -1.0, 1.0))

    ac = aco(np.sum(A * C, axis=1))
    cb = aco(np.sum(C * B, axis=1))
    on_short = np.abs((ac + cb) - theta_ab) < 1e-10

    theta_proj = aco(np.dot(C, p))       # (K,)
    theta_pa   = aco(np.dot(A, p))
    theta_pb   = aco(np.dot(B, p))
    theta_end  = np.minimum(theta_pa, theta_pb)

    return np.where(on_short, theta_proj, theta_end)

if _HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _theta_point_to_short_arc_numba(p, A, B, N, theta_ab):
        K = A.shape[0]
        out = np.empty(K, dtype=np.float64)

        for i in range(K):
            NxP = np.array([N[i,1]*p[2] - N[i,2]*p[1],
                            N[i,2]*p[0] - N[i,0]*p[2],
                            N[i,0]*p[1] - N[i,1]*p[0]])
            C = np.array([NxP[1]*N[i,2] - NxP[2]*N[i,1],
                          NxP[2]*N[i,0] - NxP[0]*N[i,2],
                          NxP[0]*N[i,1] - NxP[1]*N[i,0]])
            n = (C[0]*C[0] + C[1]*C[1] + C[2]*C[2]) ** 0.5
            if n < 1e-15:
                C = np.array([0.0, 0.0, 1.0])
            else:
                C = C / n

            def aco(x):
                if x < -1.0: x = -1.0
                elif x > 1.0: x = 1.0
                return np.arccos(x)

            ac = aco(A[i,0]*C[0] + A[i,1]*C[1] + A[i,2]*C[2])
            cb = aco(C[0]*B[i,0] + C[1]*B[i,1] + C[2]*B[i,2])
            on_short = abs((ac + cb) - theta_ab[i]) < 1e-10

            theta_proj = aco(C[0]*p[0] + C[1]*p[1] + C[2]*p[2])
            theta_pa   = aco(A[i,0]*p[0] + A[i,1]*p[1] + A[i,2]*p[2])
            theta_pb   = aco(B[i,0]*p[0] + B[i,1]*p[1] + B[i,2]*p[2])
            theta_end  = theta_pa if theta_pa < theta_pb else theta_pb
            out[i] = theta_proj if on_short else theta_end

        return out

# --------------------------- labeling core --------------------------

class BorderSampler:
    """
    Spatial labeller:
    - c1_id from country coverage (prepared geometries).
    - Nearest border via **KNN on segment midpoints** (unit vectors in 3D), then exact spherical
      distance to the short arc for a tiny candidate set (K).
    """

    def __init__(
        self,
        gpkg_path: str = GPKG_PATH,
        countries_layer: str = COUNTRIES_LAYER,
        id_field: str = ID_FIELD,
        borders_fgb: str = BORDERS_FGB,
        knn_k: int = 128,              # default candidate budget (tweakable)
        knn_expand: int = 256,         # one-shot expansion if needed
        expand_rel: float = 1.05,      # expand if kth chord <= expand_rel * best chord
    ):

        self.id_field = id_field
        self.knn_k = int(knn_k)
        self.knn_expand = int(knn_expand)
        self.expand_rel = float(expand_rel)

        # Countries (WGS84 lon/lat)
        cdf = gpd.read_file(gpkg_path, layer=countries_layer)
        if cdf.crs is None or (cdf.crs.to_epsg() or 4326) != 4326:
            cdf = cdf.to_crs(4326)
        self.countries = cdf[[id_field, "geometry"]].reset_index(drop=True)

        # Prepared geometries => fast covers()
        self._id2geom_prepared: dict[int, any] = {}
        for _, row in self.countries.iterrows():
            cid = int(row[id_field])
            self._id2geom_prepared[cid] = prep(row.geometry)

        # Borders (FlatGeobuf with segmentized shared boundaries)
        bdf = gpd.read_file(borders_fgb)
        if bdf.crs is None or (bdf.crs.to_epsg() or 4326) != 4326:
            bdf = bdf.to_crs(4326)
        self.borders = bdf.reset_index(drop=True)

        # Fast access arrays for distance loop
        self._ax = self.borders["ax"].to_numpy()
        self._ay = self.borders["ay"].to_numpy()
        self._bx = self.borders["bx"].to_numpy()
        self._by = self.borders["by"].to_numpy()
        self._id_a = self.borders["id_a"].to_numpy(dtype=int)
        self._id_b = self.borders["id_b"].to_numpy(dtype=int)

        # === Precompute segment geometry for fast vectorized distance ===
        # Endpoints as unit vectors
        A3 = lonlat_to_unitvec(self._ax, self._ay).astype(np.float64)   # (S,3)
        B3 = lonlat_to_unitvec(self._bx, self._by).astype(np.float64)   # (S,3)

        # Per-segment great-circle normal (unit)
        N3 = np.cross(A3, B3)
        N3 = _safe_div(N3, _safe_norm(N3))

        # Short-arc angle
        cos_ab = np.clip((A3 * B3).sum(axis=1), -1.0, 1.0)
        theta_ab = np.arccos(cos_ab)

        # Midpoints for KNN (normalize A+B; if degenerate (near antipodal), fall back to A)
        M3 = A3 + B3
        nn = _safe_norm(M3)
        M3 = _safe_div(M3, nn)
        near_zero = (nn[:, 0] < 1e-12)
        if np.any(near_zero):
            M3[near_zero] = A3[near_zero]

        # KD-tree on midpoints
        self._mid_tree = KDTree(M3, balanced_tree=True)
        self._A3 = A3; self._B3 = B3; self._N3 = N3
        self._theta_ab = theta_ab

    # ---- helpers

    def _country_id_at_lonlat(self, lon, lat) -> int | None:
        """Find c1_id via prepared geometry coverage (simple loop; quite fast in practice)."""
        pt = Point(lon, lat)
        for cid, pgeom in self._id2geom_prepared.items():
            if pgeom.covers(pt):
                return cid
        return None

    def _knn_candidate_indices(self, p: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        KNN in 3D on unit midpoints. Returns (idx, chord_dist).
        For unit vectors, chord distance d relates to angular distance α via d = 2 sin(α/2).
        """
        d, idx = self._mid_tree.query(p, k=min(k, self._mid_tree.n), workers=1)
        if np.isscalar(d):  # normalize to 1D arrays
            d = np.array([d], dtype=np.float64); idx = np.array([idx], dtype=np.int64)
        return idx.astype(np.int64), d.astype(np.float64)

    def _nearest_segment_vectorized(self, lon: float, lat: float) -> tuple[float, int, int]:
        """
        Vectorized nearest short-arc distance using precomputed A3, B3, N3 and KDTree shortlist.
        Adaptive expansion: if kth chord is ~as small as the best chord, re-query with 2K once.
        """
        p = lonlat_to_unitvec(np.array([lon], dtype=np.float64),
                              np.array([lat], dtype=np.float64))[0]  # (3,)

        idx, chord = self._knn_candidate_indices(p, self.knn_k)
        A = self._A3[idx]; B = self._B3[idx]; N = self._N3[idx]; th = self._theta_ab[idx]

        if _HAS_NUMBA:
            theta = _theta_point_to_short_arc_numba(p, A, B, N, th)
        else:
            theta = _theta_point_to_short_arc_numpy(p, A, B, N, th)

        best_i = int(np.argmin(theta))
        best_theta = float(theta[best_i])
        best_idx = int(idx[best_i])
        best_d = float(R_EARTH_KM * best_theta)

        # heuristic expansion
        kth = float(np.max(chord))  # coarse proxy; SciPy usually returns sorted
        if kth <= self.expand_rel * float(chord[best_i]) and self.knn_expand > self.knn_k:
            idx2, _ = self._knn_candidate_indices(p, self.knn_expand)
            uni = np.unique(idx2)
            A = self._A3[uni]; B = self._B3[uni]; N = self._N3[uni]; th = self._theta_ab[uni]
            if _HAS_NUMBA:
                theta2 = _theta_point_to_short_arc_numba(p, A, B, N, th)
            else:
                theta2 = _theta_point_to_short_arc_numpy(p, A, B, N, th)
            b2 = int(np.argmin(theta2))
            if float(theta2[b2]) < best_theta:
                best_theta = float(theta2[b2])
                best_idx = int(uni[b2])
                best_d = float(R_EARTH_KM * best_theta)

        ida = int(self._id_a[best_idx]); idb = int(self._id_b[best_idx])
        return best_d, ida, idb

    # ---- public labeling

    def sample_lonlat(self, lon: float, lat: float) -> tuple[float, int, int]:
        """
        Return (distance_km [unsigned], c1_id, c2_id) at (lon,lat).
        c2_id is the *other* country of the nearest border segment.
        """
        c1 = self._country_id_at_lonlat(lon, lat)
        best_d, a, b = self._nearest_segment_vectorized(lon, lat)

        if c1 == a:
            c2 = b
        elif c1 == b:
            c2 = a
        else:
            c2 = b  # rare corner cases (tiny islands, topology quirks)

        return float(best_d), int(c1) if c1 is not None else -1, int(c2)

# --------------------------- near-border drawing --------------------

def _draw_near_border_points(sampler: BorderSampler, M: int, rng: np.random.Generator):
    """
    Draw M points near borders by sampling segments ~ angular length, SLERP to a random point,
    then offset by a small random geodesic distance into a random side.
    Returns: lon, lat, xyz, is_border(=1), r_band, d_km_hint, id_a, id_b
    """
    A = sampler._A3
    B = sampler._B3
    
    # 1) pick segments proportional to arc length
    theta = sampler._theta_ab
    # mask out tiny/degenerate arcs up front
    tiny = theta < 1e-9
    probs = np.where(tiny, 0.0, theta) # probability distribution based on arc length
    probs = probs / probs.sum() # make the probability distribution sum to 1
    seg_idx = rng.choice(len(theta), size=M, p=probs) # randomly choose segments based on the probability
    A = sampler._A3[seg_idx]            # (M,3)
    B = sampler._B3[seg_idx]            # (M,3)
    N = sampler._N3[seg_idx]            # (M,3)  unit normals
    th = sampler._theta_ab[seg_idx]     # (M,)
    
    # 2) uniform SLERP along the short arc
    t = rng.random(M)
    sin_th = np.sin(th)
    small = sin_th < 1e-12
    coef_a = np.empty(M)
    coef_b = np.empty(M)
    coef_a[~small] = np.sin((1.0 - t[~small]) * th[~small]) / sin_th[~small]
    coef_b[~small] = np.sin(t[~small] * th[~small]) / sin_th[~small]
    # linear fallback when arc is extremely small
    coef_a[small] = 1.0 - t[small]
    coef_b[small] = t[small]

    p = (A * coef_a[:, None] + B * coef_b[:, None])
    p = _safe_div(p, _safe_norm(p))

    # 3) local frame at p using N: tangent along border and outward normal in tangent plane
    tgc = np.cross(N, p)                             # tangent along border
    tgc = _safe_div(tgc, _safe_norm(tgc))
    u = np.cross(tgc, p)                             # perp to border in tangent plane
    u = _safe_div(u, _safe_norm(u))
    side = rng.integers(0, 2, size=M) * 2 - 1        # ±1
    u = u * side[:, None]
    
    # 4) sample log-uniform offset bands
    d_km_hint, r_band = _sample_distance_km(M, rng)  
    theta_off = (d_km_hint.astype(np.float64) / R_EARTH_KM)

    # 5) geodesic move: p_off = p*cos(θ) + u*sin(θ)
    c = np.cos(theta_off)[:, None]; s = np.sin(theta_off)[:, None]
    p_off = p * c + u * s
    p_off = _safe_div(p_off, _safe_norm(p_off))

    lon, lat = unitvec_to_lonlat(p_off)
    xyz = p_off.astype(np.float32)
    is_border = np.ones(M, np.uint8)
    id_a = sampler._id_a[seg_idx].astype(np.int32)
    id_b = sampler._id_b[seg_idx].astype(np.int32)
    
    return lon, lat, xyz, is_border, r_band.astype(np.uint8), d_km_hint.astype(np.float32), id_a, id_b

# --------------------------- multiprocessing ------------------------

# Globals living inside workers (one-time init per process)
_BS: BorderSampler | None = None
_PID = None
_PROCNAME = None

def _init_worker(gpkg_path, layer, id_field, borders_fgb, knn_k, knn_expand, expand_rel):
    """Build spatial indices ONCE per worker (GEOS/GDAL/KD-tree)."""
    global _BS, _PID, _PROCNAME
    _BS = BorderSampler(
        gpkg_path, layer, id_field, borders_fgb,
        knn_k=knn_k, knn_expand=knn_expand, expand_rel=expand_rel
    )
    import multiprocessing as _mp
    _PID = os.getpid()
    _PROCNAME = _mp.current_process().name
    
def _label_and_write_chunk_with_full_tree(args):
    """
    Label a chunk and write a TEMP Parquet shard.
    For all points, it always uses the KDTree search.
    Stats: counts (fast/full), polygon covers time, distance time, write time.
    """
    import time
    t0 = time.perf_counter()
    try:
        (gpkg_path, layer, id_field, borders_fgb,
         lon_chunk, lat_chunk, xyz_chunk, is_border_chunk, r_band_chunk,
         dkm_hint_chunk, ida_chunk, idb_chunk,
         temp_path, chunk_idx, knn_k, knn_expand, expand_rel) = args

        global _BS, _PID, _PROCNAME
        if _BS is None:
            _init_worker(gpkg_path, layer, id_field, borders_fgb, knn_k, knn_expand, expand_rel)

        stats = {"points": int(len(lon_chunk)), "fast": 0, "full": 0,
                 "poly_query_s": 0.0, "dist_s": 0.0, "write_s": 0.0}

        N = len(lon_chunk)
        dist = np.empty(N, dtype=np.float32)
        c1   = np.empty(N, dtype=np.int32)
        c2   = np.empty(N, dtype=np.int32)

        mb = (is_border_chunk == 1)

        # Always-compute path: ignore hints and is_border flag for labeling.
        # Compute true nearest-border distance and (c1,c2) for ALL rows.
        stats["full"] += int(N)

        for i in range(N):
            lon = float(lon_chunk[i]); lat = float(lat_chunk[i])

            # 1) nearest border distance + border pair (a,b)
            t0d = time.perf_counter()
            d_km, a, b = _BS._nearest_segment_vectorized(lon, lat)
            stats["dist_s"] += (time.perf_counter() - t0d)
            dist[i] = d_km

            # 2) decide (c1,c2)
            # Try prepared polygon covers first (fast when available)
            pt = Point(lon, lat)
            pgeom_a = _BS._id2geom_prepared.get(a)
            pgeom_b = _BS._id2geom_prepared.get(b)

            decided = False
            t0p = time.perf_counter()
            if pgeom_a is not None and pgeom_a.covers(pt):
                c1[i], c2[i] = a, b
                decided = True
            elif pgeom_b is not None and pgeom_b.covers(pt):
                c1[i], c2[i] = b, a
                decided = True
            stats["poly_query_s"] += (time.perf_counter() - t0p)

            if not decided:
                # Fallback: global country lookup at point
                c1_id = _BS._country_id_at_lonlat(lon, lat)
                if c1_id is not None:
                    c1[i] = int(c1_id)
                    # Choose c2 as the "other" of the nearest pair if possible
                    if c1[i] == a:
                        c2[i] = b
                    elif c1[i] == b:
                        c2[i] = a
                    else:
                        # If the containing country isn't one of the nearest pair (rare),
                        # keep nearest pair order (a,b) as the adjacency label.
                        c2[i] = b
                else:
                    # Last resort: use nearest pair order
                    c1[i], c2[i] = a, b

        # Write shard
        t_w0 = time.perf_counter()
        df = pd.DataFrame({
            "lon": lon_chunk.astype(np.float32),
            "lat": lat_chunk.astype(np.float32),
            "x": xyz_chunk[:,0].astype(np.float32),
            "y": xyz_chunk[:,1].astype(np.float32),
            "z": xyz_chunk[:,2].astype(np.float32),
            "dist_km": dist,
            "log1p_dist": np.log1p(dist).astype(np.float32),
            "c1_id": c1,
            "c2_id": c2,
            "is_border": is_border_chunk.astype(np.uint8),
            "r_band": r_band_chunk.astype(np.uint8),
        })
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, temp_path, compression="zstd")
        stats["write_s"] += (time.perf_counter() - t_w0)

        elapsed = time.perf_counter() - t0
        print(f"Chunk {chunk_idx} of size {N} finished by process {_PID} aka {_PROCNAME}: {elapsed:.3f}s")
        return ("ok", {"path": temp_path, "pid": _PID, "proc": _PROCNAME,
                       "elapsed": elapsed, "rows": int(N), "stats": stats})
    except Exception as e:
        import traceback
        return ("err", {"error": str(e), "traceback": traceback.format_exc()})
    

def _label_and_write_chunk(args):
    """
    Label a chunk and write a TEMP Parquet shard.
    Stats: counts (fast/full), polygon covers time, distance time, write time.
    """
    import time
    t0 = time.perf_counter()
    try:
        (gpkg_path, layer, id_field, borders_fgb,
         lon_chunk, lat_chunk, xyz_chunk, is_border_chunk, r_band_chunk,
         dkm_hint_chunk, ida_chunk, idb_chunk,
         temp_path, chunk_idx, knn_k, knn_expand, expand_rel) = args

        global _BS, _PID, _PROCNAME
        if _BS is None:
            _init_worker(gpkg_path, layer, id_field, borders_fgb, knn_k, knn_expand, expand_rel)

        stats = {"points": int(len(lon_chunk)), "fast": 0, "full": 0,
                 "poly_query_s": 0.0, "dist_s": 0.0, "write_s": 0.0}

        N = len(lon_chunk)
        dist = np.empty(N, dtype=np.float32)
        c1   = np.empty(N, dtype=np.int32)
        c2   = np.empty(N, dtype=np.int32)

        mb = (is_border_chunk == 1)

        # Near-border fast path: trust offset magnitude, decide side by prepared covers
        if np.any(mb):
            idxs = np.where(mb)[0]
            stats["fast"] += int(idxs.size)
            dist[mb] = dkm_hint_chunk[mb]
            for i in idxs:
                lon = float(lon_chunk[i]); lat = float(lat_chunk[i])
                ida = int(ida_chunk[i]); idb = int(idb_chunk[i])
                pgeom_a = _BS._id2geom_prepared.get(ida)
                pgeom_b = _BS._id2geom_prepared.get(idb)
                pt = Point(lon, lat)

                t0p = time.perf_counter()
                decided = False
                if pgeom_a is not None and pgeom_a.covers(pt):
                    c1[i], c2[i] = ida, idb
                    decided = True
                elif pgeom_b is not None and pgeom_b.covers(pt):
                    c1[i], c2[i] = idb, ida
                    decided = True
                stats["poly_query_s"] += (time.perf_counter() - t0p)

                # Default: trust the hint
                d_hint = float(dkm_hint_chunk[i])
                use_hint = True

                if VALIDATE_NEAR_BORDER:
                    # One KNN nearest-segment check
                    t0d = time.perf_counter()
                    d_knn, a_knn, b_knn = _BS._nearest_segment_vectorized(lon, lat)
                    stats["dist_s"] += (time.perf_counter() - t0d)

                    # If nearest border pair doesn't match (ida,idb) in either order,
                    # or the computed distance is much smaller than the hint,
                    # override with computed nearest.
                    same_pair = ((a_knn == ida and b_knn == idb) or (a_knn == idb and b_knn == ida))
                    if (not same_pair) or (d_knn < HINT_DELTA_FACTOR * d_hint):
                        dist[i] = d_knn
                        # If polygon side is unknown or inconsistent, go with nearest pair.
                        if not decided:
                            c1[i], c2[i] = a_knn, b_knn
                        use_hint = False
                    else:
                        # keep hint distance; ids already set if decided, else pick side by nearest pair
                        if not decided:
                            # If we didn't settle side via polygons, choose the country from nearest pair
                            # that contains the point (rare). If neither contains, default to nearest pair order.
                            if _BS._id2geom_prepared.get(a_knn, None) and _BS._id2geom_prepared[a_knn].covers(pt):
                                c1[i], c2[i] = a_knn, b_knn
                            elif _BS._id2geom_prepared.get(b_knn, None) and _BS._id2geom_prepared[b_knn].covers(pt):
                                c1[i], c2[i] = b_knn, a_knn
                            else:
                                c1[i], c2[i] = a_knn, b_knn

                if use_hint:
                    dist[i] = d_hint

        # Uniform full path
        mu = ~mb
        if np.any(mu):
            idxs = np.where(mu)[0]
            stats["full"] += int(idxs.size)
            for i in idxs:
                lon = float(lon_chunk[i]); lat = float(lat_chunk[i])
                t0d = time.perf_counter()
                d_km, a, b = _BS._nearest_segment_vectorized(lon, lat)
                stats["dist_s"] += (time.perf_counter() - t0d)

                dist[i] = d_km
                c1_id = _BS._country_id_at_lonlat(lon, lat)
                if c1_id is None:
                    c1[i], c2[i] = (a, b)
                else:
                    c1[i] = int(c1_id)
                    c2[i] = b if c1[i] == a else a

        # Write shard
        t_w0 = time.perf_counter()
        df = pd.DataFrame({
            "lon": lon_chunk.astype(np.float32),
            "lat": lat_chunk.astype(np.float32),
            "x": xyz_chunk[:,0].astype(np.float32),
            "y": xyz_chunk[:,1].astype(np.float32),
            "z": xyz_chunk[:,2].astype(np.float32),
            "dist_km": dist,
            "log1p_dist": np.log1p(dist).astype(np.float32),
            "c1_id": c1,
            "c2_id": c2,
            "is_border": is_border_chunk.astype(np.uint8),
            "r_band": r_band_chunk.astype(np.uint8),
        })
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, temp_path, compression="zstd")
        stats["write_s"] += (time.perf_counter() - t_w0)

        elapsed = time.perf_counter() - t0
        print(f"Chunk {chunk_idx} of size {N} finished by process {_PID} aka {_PROCNAME}: {elapsed:.3f}s")
        return ("ok", {"path": temp_path, "pid": _PID, "proc": _PROCNAME,
                       "elapsed": elapsed, "rows": int(N), "stats": stats})
    except Exception as e:
        import traceback
        return ("err", {"error": str(e), "traceback": traceback.format_exc()})

# --------------------------- concat utility -------------------------

def _concat_parquet_shards(shard_paths: list[str], out_path: str,
                           compression: str = "zstd", row_group_size: int | None = 512_000):
    """Concatenate shards into one Parquet by streaming row-groups (stable, memory-bounded)."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    first_schema = None
    writer = None
    try:
        for p in shard_paths:
            pf = pq.ParquetFile(p)
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                if first_schema is None:
                    first_schema = table.schema
                    writer = pq.ParquetWriter(out_path, first_schema, compression=compression)
                writer.write_table(table, row_group_size=row_group_size)
            del pf
    finally:
        if writer is not None:
            writer.close()

# --------------------------- public API -----------------------------

def make_dataset_parallel(
    n_total: int,
    out_path: str = os.path.join(FOLDER_PATH, "dataset_all.parquet"),
    mixture = (0.70, 0.30),          # (near-border, uniform)
    shards_per_total: int = 24,
    max_workers: int | None = None,
    seed: int | None = 42,
    gpkg_path: str = GPKG_PATH,
    countries_layer: str = COUNTRIES_LAYER,
    id_field: str = ID_FIELD,
    borders_fgb: str = BORDERS_FGB,
    tmp_subdir: str = "tmp_shards",
    writer_compression: str = "zstd",
    writer_row_group_size: int | None = 512_000,
    points_path: str | None = None,         # load starting points (apples-to-apples)
    export_points_path: str | None = None,  # save sampled points (for reuse)
    shuffle_points: bool = True,            # False -> keep exact order
    # KNN knobs (propagate to workers)
    knn_k: int = 128,
    knn_expand: int = 256,
    expand_rel: float = 1.05,
    reliable: bool = False # Use full tree for all points, not just the uniformly sampled points
) -> str:
    """
    Generate n_total samples and write ONE Parquet at `out_path`.
    Returns the absolute path to the written Parquet file.
    """

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_dir = os.path.dirname(out_path) or "."
    tmp_dir = os.path.join(out_dir, tmp_subdir)
    os.makedirs(tmp_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    # 1) Draw or load points (parent process only; no labeling here)
    if points_path is not None:
        # TODO: make it work with log1p_dist
        t0 = time.perf_counter()
        (lon, lat, xyz, is_border, r_band, dkm_hint, ida, idb) = _load_points_parquet(points_path)
        print(f"Loaded starting points from '{points_path}' in {time.perf_counter() - t0:.3f}s")
        if shuffle_points:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(lon))
            lon, lat, xyz, is_border, r_band, dkm_hint, ida, idb = \
                lon[perm], lat[perm], xyz[perm], is_border[perm], r_band[perm], dkm_hint[perm], ida[perm], idb[perm]
    else:
        t0 = time.perf_counter()
        sampler_for_drawing = BorderSampler(
            gpkg_path=gpkg_path,
            countries_layer=countries_layer,
            id_field=id_field,
            borders_fgb=borders_fgb,
            knn_k=knn_k, knn_expand=knn_expand, expand_rel=expand_rel,
        )
        print(f"Sampler creation: {time.perf_counter() - t0:.3f}s")
        t0 = time.perf_counter()

        n_border = int(mixture[0] * n_total)
        n_uniform = n_total - n_border

        # sample n_border points close to the borders
        (lon_b, lat_b, xyz_b, is_b_b, r_band_b, dkm_b, ida_b, idb_b) = \
            _draw_near_border_points(sampler_for_drawing, n_border, rng)

        # sample n_uniform points uniformly across the globe
        xyz_u = rng.normal(size=(n_uniform, 3)).astype(np.float64)
        xyz_u = _safe_div(xyz_u, _safe_norm(xyz_u)).astype(np.float32)
        lon_u, lat_u = unitvec_to_lonlat(xyz_u)
        is_b_u = np.zeros(n_uniform, np.uint8)
        r_band_u = np.full(n_uniform, 255, np.uint8)  # 255 marks 'uniform' bucket
        dkm_u = np.zeros(n_uniform, np.float32)
        ida_u = np.zeros(n_uniform, np.int32)
        idb_u = np.zeros(n_uniform, np.int32)

        lon = np.concatenate([lon_b, lon_u]).astype(np.float32)
        lat = np.concatenate([lat_b, lat_u]).astype(np.float32)
        xyz = np.vstack([xyz_b, xyz_u]).astype(np.float32)
        is_border = np.concatenate([is_b_b, is_b_u])
        r_band    = np.concatenate([r_band_b, r_band_u])
        dkm_hint  = np.concatenate([dkm_b, dkm_u]).astype(np.float32)
        ida       = np.concatenate([ida_b, ida_u]).astype(np.int32)
        idb       = np.concatenate([idb_b, idb_u]).astype(np.int32)

        if shuffle_points:
            perm = rng.permutation(len(lon))
            lon, lat, xyz, is_border, r_band, dkm_hint, ida, idb = \
                lon[perm], lat[perm], xyz[perm], is_border[perm], r_band[perm], dkm_hint[perm], ida[perm], idb[perm]

        if export_points_path:
            ep = _save_points_parquet(export_points_path, lon, lat, xyz, is_border, r_band, dkm_hint, ida, idb)
            print(f"Exported starting points to '{ep}'")

    print(f"Parent Processes: {time.perf_counter() - t0:.3f}s")
    t0 = time.perf_counter()
    
    # 2) Parallel labeling into many temp shards
    total = len(lon)
    shards = max(1, int(shards_per_total))
    chunk_size = max(1, int(np.ceil(total / shards)))

    if max_workers is None:
        try:
            max_workers = max(1, (os.cpu_count() or 4) - 1)
        except Exception:
            max_workers = 4

    futures, temp_paths = [], []
    agg = defaultdict(float)
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(gpkg_path, countries_layer, id_field, borders_fgb, knn_k, knn_expand, expand_rel),
    ) as ex:
        for ci in range(shards):
            lo = ci * chunk_size
            hi = min((ci + 1) * chunk_size, total)
            if lo >= hi:
                continue
            tmp_path = os.path.join(tmp_dir, f"part-{ci:05d}.parquet")
            args = (
                gpkg_path, countries_layer, id_field, borders_fgb,
                lon[lo:hi], lat[lo:hi], xyz[lo:hi],
                is_border[lo:hi], r_band[lo:hi],
                dkm_hint[lo:hi], ida[lo:hi], idb[lo:hi],
                tmp_path, ci, knn_k, knn_expand, expand_rel
            )
            if reliable: futures.append(ex.submit(_label_and_write_chunk_with_full_tree, args))
            else: futures.append(ex.submit(_label_and_write_chunk, args))

        it = as_completed(futures)
        if tqdm: it = tqdm(it, total=len(futures), desc="Labeling shards")
        for f in it:
            status, payload = f.result()
            if status == "err":
                raise RuntimeError(f"Worker failed: {payload['error']}\n{payload['traceback']}")
            info = payload
            temp_paths.append(info["path"])

            s = info.get("stats", {})
            agg["points"]       += s.get("points", 0)
            agg["fast"]         += s.get("fast", 0)
            agg["full"]         += s.get("full", 0)
            agg["poly_query_s"] += s.get("poly_query_s", 0.0)
            agg["dist_s"]       += s.get("dist_s", 0.0)
            agg["write_s"]      += s.get("write_s", 0.0)

    print(
        "[aggregate] "
        f"points={int(agg['points'])} fast={int(agg['fast'])} full={int(agg['full'])}  "
        f"t_poly={agg['poly_query_s']:.1f}s t_dist={agg['dist_s']:.1f}s t_write={agg['write_s']:.1f}s"
    )

    print(f"Parallel labeling: {time.perf_counter() - t0:.3f}s")
    t0 = time.perf_counter()

    # 3) Concatenate shards -> ONE Parquet
    _concat_parquet_shards(temp_paths, out_path, compression=writer_compression,
                           row_group_size=writer_row_group_size)

    # 4) Cleanup
    for p in temp_paths:
        try: os.remove(p)
        except Exception: pass
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass
    
    print(f"Concatenation and Clean up: {time.perf_counter() - t0:.3f}s")
    return os.path.abspath(out_path)

def query_point(
    lon: float,
    lat: float,
    gpkg_path: str = GPKG_PATH,
    countries_layer: str = COUNTRIES_LAYER,
    id_field: str = ID_FIELD,
    borders_fgb: str = BORDERS_FGB,
):
    """
    Print the labeling result for a single (lon, lat).
    """
    bs = BorderSampler(
        gpkg_path=gpkg_path,
        countries_layer=countries_layer,
        id_field=id_field,
        borders_fgb=borders_fgb,
    )
    d_km, c1, c2 = bs.sample_lonlat(float(lon), float(lat))
    print(f"== lon: {float(lon):.6f}, lat: {float(lat):.6f}, dist_km: {d_km:.6f}, c1: {c1}, c2: {c2} ==")

# --------------------------- execution -------------------------

def run():
    # safety for multiprocessing
    mp.set_start_method("spawn", force=True)

    t0 = time.perf_counter()
    path = make_dataset_parallel(
        n_total=10_000_000,
        out_path=os.path.join(FOLDER_PATH, "parquet/log_dataset_10M.parquet"),
        mixture=(0.70, 0.30),
        shards_per_total=32,
        max_workers=None,
        seed=None,
        knn_k=128,
        knn_expand=256,
        expand_rel=1.05,
        reliable=True
    )
    dt = time.perf_counter() - t0
    print(f"Total time Elapsed: {dt:.3f}s")
    
    
START_POINTS = os.path.join(FOLDER_PATH, "parquet/start_points_10k.parquet")
def run_det():
    # safety for multiprocessing
    mp.set_start_method("spawn", force=True)

    t0 = time.perf_counter()
    path = make_dataset_parallel(
        n_total=10_000_000,
        out_path=os.path.join(FOLDER_PATH, "parquet/sampler_10M.parquet"),
        mixture=(0.70, 0.30),
        shards_per_total=32,
        max_workers=None,
        seed=37, 
    )
    dt = time.perf_counter() - t0
    print(f"Total time Elapsed: {dt:.3f}s")

if __name__ == "__main__":
    run()
    #query_point(6.225000, 50.555400)  # Vennbahn, Germany/Belgium
