"""
Minimal, production-ready generator for (lon, lat, x, y, z, dist_km, c1_id, c2_id, is_border, r_band)
saved as **one Parquet file**.

Major design choices:
- We bias sampling near borders (70/30 by default).
- Distances are **spherical** to short-arc border segments.
- Labeling is parallel: each worker builds its own spatial indices once, then writes a temp shard.
  The parent concatenates shards into **one Parquet file** (stable, memory-bounded).
- NEW (perf): Segment shortlist via **KNN on segment midpoints in 3D** (cKDTree), not STRtree buffers.
  This bounds candidates to O(K) per point (K≈64) and vectorizes the distance kernel.
- NEW (perf): Optional **Numba** JIT for the distance kernel (falls back to NumPy if unavailable).
- Windows/macOS-safe (spawn start method, __main__ guard).
"""

from __future__ import annotations

# --- stdlib
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# --- third-party
import numpy as np
import pandas as pd
import geopandas as gpd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Point
from shapely.prepared import prep

try:
    # fast KNN over unit vectors (midpoints)
    from scipy.spatial import cKDTree
except Exception as _e:  # give a nicer error message later if user calls make_dataset_parallel
    cKDTree = None

try:
    # optional progress bar; gracefully degrades to range()
    from tqdm import tqdm
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

# Inputs you already have
GPKG_PATH   = os.path.join(FOLDER_PATH, "world_bank_geodata.gpkg")
BORDERS_FGB = os.path.join(FOLDER_PATH, "borders.fgb")
COUNTRIES_LAYER = "countries"
ID_FIELD    = "id"                     # numeric country id in your GPKG


# --------------------------- small utilities ------------------------

def _safe_norm(v: np.ndarray, axis=1, keepdims=True, eps=1e-15):
    """Avoid division by ~0 when normalizing tiny vectors."""
    n = np.linalg.norm(v, axis=axis, keepdims=keepdims)
    return np.where(n < eps, eps, n)

def _safe_div(v: np.ndarray, n: np.ndarray, eps=1e-15):
    """Guarded division used across spherical ops."""
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

def greatcircle_point_segment_dist_km(p_lon, p_lat, a_lon, a_lat, b_lon, b_lat) -> float:
    """
    Scalar reference implementation (kept for tests).
    Unsigned spherical distance (km) from P to the **short** arc A-B.
    """
    def _ll2v(lon, lat):
        lon = np.radians(lon); lat = np.radians(lat)
        cl = np.cos(lat)
        return np.array([cl*np.cos(lon), cl*np.sin(lon), np.sin(lat)], dtype=np.float64)

    p = _ll2v(p_lon, p_lat)
    a = _ll2v(a_lon, a_lat)
    b = _ll2v(b_lon, b_lat)

    n = np.cross(a, b); nn = np.linalg.norm(n)
    if nn == 0.0:
        # identical endpoints -> min distance to endpoints
        return R_EARTH_KM * min(
            np.arccos(np.clip(np.dot(p, a), -1.0, 1.0)),
            np.arccos(np.clip(np.dot(p, b), -1.0, 1.0)),
        )
    n /= nn

    # project foot of perpendicular from P onto the great circle (A,B)
    c = np.cross(n, p); c = np.cross(c, n); c /= np.linalg.norm(c)

    ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    ac = np.arccos(np.clip(np.dot(a, c), -1.0, 1.0))
    cb = np.arccos(np.clip(np.dot(c, b), -1.0, 1.0))

    # If C lies on the short arc, distance is P-C; else min endpoint distance.
    if abs((ac + cb) - ab) < 1e-10:
        theta = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
    else:
        theta = min(
            np.arccos(np.clip(np.dot(p, a), -1.0, 1.0)),
            np.arccos(np.clip(np.dot(p, b), -1.0, 1.0)),
        )
    return float(R_EARTH_KM * theta)

# --------------------------- sampling logic -------------------------

def _sample_distance_km(batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Piecewise log-uniform bands, favoring near-border points:
    [0-10] km (60%), [10-50] (30%), [50-300] (10%).
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
    # projection of p to each GC(A,B): C = norm( (N x p) x N )
    C = np.cross(N, p)                   # (K,3)
    C = np.cross(C, N)                   # (K,3)
    C = _safe_div(C, _safe_norm(C))      # (K,3)

    # helper
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
            # C = normalize( (N_i x p) x N_i )
            NxP = np.array([N[i,1]*p[2] - N[i,2]*p[1],
                            N[i,2]*p[0] - N[i,0]*p[2],
                            N[i,0]*p[1] - N[i,1]*p[0]])
            C = np.array([NxP[1]*N[i,2] - NxP[2]*N[i,1],
                          NxP[2]*N[i,0] - NxP[0]*N[i,2],
                          NxP[0]*N[i,1] - NxP[1]*N[i,0]])
            # normalize C
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
      distance to the short arc for a tiny candidate set (K≈64).
    Why this design?
      STRtree+buffer returns *thousands* of segments in dense areas, exploding the distance loop.
      KNN bounds candidates deterministically and lets us vectorize/JIT the kernel.
    """

    def __init__(
        self,
        gpkg_path: str = GPKG_PATH,
        countries_layer: str = COUNTRIES_LAYER,
        id_field: str = ID_FIELD,
        borders_fgb: str = BORDERS_FGB,
        knn_k: int = 64,              # default candidate budget
        knn_expand: int = 128,        # one-shot expansion if needed
        expand_rel: float = 1.05,     # expand if kth chord <= expand_rel * best chord
    ):
        if cKDTree is None:
            raise RuntimeError("scipy.spatial.cKDTree is required. `pip install scipy`.")

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
        for ridx, row in self.countries.iterrows():
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

        # Midpoints for KNN (normalize A+B; if degenerate, fall back to A)
        M3 = A3 + B3
        nn = _safe_norm(M3)
        M3 = _safe_div(M3, nn)
        # For near-antipodal (A≈-B) segments, A+B ~ 0; fallback to A
        near_zero = (nn[:, 0] < 1e-12)
        if np.any(near_zero):
            M3[near_zero] = A3[near_zero]

        # Build KD-tree on midpoints
        self._mid_tree = cKDTree(M3, balanced_tree=True)
        self._A3 = A3; self._B3 = B3; self._N3 = N3
        self._theta_ab = theta_ab

    # ---- helpers

    def _country_id_at_lonlat(self, lon, lat) -> int | None:
        """Find c1_id via prepared geometry coverage."""
        pt = Point(lon, lat)
        # Small optimization: try all candidates whose bbox intersects the point would require STRtree,
        # but prepared.covers(pt) is already fast; just check all is fine for this call frequency.
        # If this ever shows up in profiles, we can add a polygon STRtree + prepared filter.
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
        # SciPy returns scalars when k==1; normalize to 1D arrays
        if np.isscalar(d):
            d = np.array([d], dtype=np.float64); idx = np.array([idx], dtype=np.int64)
        return idx.astype(np.int64), d.astype(np.float64)

    def _nearest_segment_vectorized(self, lon: float, lat: float) -> tuple[float, int, int]:
        """
        Vectorized nearest short-arc distance using precomputed A3, B3, N3 and KDTree shortlist.
        Adaptive expansion: if kth chord is ~as small as the best chord, re-query with 2K once.
        """
        # Point as unit vector
        p = lonlat_to_unitvec(np.array([lon], dtype=np.float64),
                              np.array([lat], dtype=np.float64))[0]  # (3,)

        # First pass (K)
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

        # Heuristic: if kth chord ≈ best chord, expand to 2K once to de-risk misses
        # chord and angle are monotonic, so this works as a conservative guard.
        kth = float(np.max(chord))  # last of K (SciPy doesn't guarantee sorted, but typical)
        if kth <= self.expand_rel * float(chord[best_i]) and self.knn_expand > self.knn_k:
            idx2, chord2 = self._knn_candidate_indices(p, self.knn_expand)
            # merge unique
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
            # Rare tiny-island/topology corner case: just return the nearest pair.
            c2 = b

        return float(best_d), int(c1) if c1 is not None else -1, int(c2)

# --------------------------- near-border drawing --------------------

def _draw_near_border_points(sampler: BorderSampler, M: int, rng: np.random.Generator):
    """
    Draw M points near borders:
    1) choose segments proportional to angular length,
    2) SLERP along segment to a random point,
    3) offset by a random small geodesic distance into one side (±).

    Returns:
      lon, lat, xyz, is_border(=1), r_band, d_km_hint, id_a, id_b
    """
    ax = sampler._ax.astype(np.float64); ay = sampler._ay.astype(np.float64)
    bx = sampler._bx.astype(np.float64); by = sampler._by.astype(np.float64)
    A  = lonlat_to_unitvec(ax, ay).astype(np.float64)
    B  = lonlat_to_unitvec(bx, by).astype(np.float64)

    # Remove near-degenerate segments (numerically unstable + useless)
    dot_all = np.clip((A * B).sum(axis=1), -1.0, 1.0)
    theta_seg_all = np.arccos(dot_all)
    valid = theta_seg_all > 1e-9
    A, B   = A[valid], B[valid]
    ida_all = sampler._id_a[valid].astype(np.int32)
    idb_all = sampler._id_b[valid].astype(np.int32)

    # Sample segments ~ proportional to angular span
    theta_seg = theta_seg_all[valid] + 1e-15
    probs = (theta_seg / theta_seg.sum()).astype(np.float64)
    idx = rng.choice(len(A), size=M, p=probs)

    a = A[idx]; b = B[idx]
    # SLERP to random point along each segment
    dot_ab = np.clip((a * b).sum(axis=1), -1.0, 1.0)
    theta  = np.arccos(dot_ab)
    t = rng.random(M)
    sin_th = np.sin(theta)
    small = sin_th < 1e-12
    coef_a = np.empty_like(theta); coef_b = np.empty_like(theta)
    coef_a[~small] = np.sin((1.0 - t[~small]) * theta[~small]) / sin_th[~small]
    coef_b[~small] = np.sin(t[~small] * theta[~small]) / sin_th[~small]
    coef_a[small]  = 1.0 - t[small]; coef_b[small] = t[small]
    p = a * coef_a[:, None] + b * coef_b[:, None]
    p = _safe_div(p, _safe_norm(p))

    # Tangent basis + choose a side (±) to step away from the border
    n = np.cross(a, b); n = _safe_div(n, _safe_norm(n))
    tgc = np.cross(n, p); tgc = _safe_div(tgc, _safe_norm(tgc))
    u = np.cross(tgc, p); u = _safe_div(u, _safe_norm(u))
    u *= np.where(rng.random(M) < 0.5, 1.0, -1.0)[:, None]

    # Distance hints (km) so the worker can fast-path for most near-border points
    d_km_hint, r_band = _sample_distance_km(M, rng)
    theta_off = (d_km_hint.astype(np.float64) / R_EARTH_KM)
    p_off = move_along_geodesic(p.astype(np.float32), u.astype(np.float32), theta_off.astype(np.float32))

    lon, lat = unitvec_to_lonlat(p_off)
    xyz = p_off.astype(np.float32)
    is_border = np.ones(M, np.uint8)
    id_a = ida_all[idx]; id_b = idb_all[idx]
    return lon, lat, xyz, is_border, r_band, d_km_hint.astype(np.float32), id_a, id_b

# --------------------------- multiprocessing ------------------------

# Globals living inside workers (one-time init per process)
_BS: BorderSampler | None = None
_PID = None
_PROCNAME = None

def _init_worker(gpkg_path, layer, id_field, borders_fgb):
    """
    Build spatial indices ONCE per worker (heavy GEOS/GDAL/trees).
    """
    global _BS, _PID, _PROCNAME
    _BS = BorderSampler(gpkg_path, layer, id_field, borders_fgb)
    import multiprocessing as _mp
    _PID = os.getpid()
    _PROCNAME = _mp.current_process().name

def _label_and_write_chunk(args):
    """
    Label a chunk and write a TEMP Parquet shard.

    Instrumentation:
      - counts: points, fast (near-border) vs full (uniform) paths
      - timings: polygon covers, distance kernel, parquet write
    """
    import time
    t0 = time.perf_counter()
    try:
        (gpkg_path, layer, id_field, borders_fgb,
         lon_chunk, lat_chunk, xyz_chunk, is_border_chunk, r_band_chunk,
         dkm_hint_chunk, ida_chunk, idb_chunk,
         temp_path, chunk_idx) = args

        global _BS, _PID, _PROCNAME
        if _BS is None:
            _init_worker(gpkg_path, layer, id_field, borders_fgb)

        # ---------- stats ----------
        stats = {
            "points": int(len(lon_chunk)),
            "fast": 0,
            "full": 0,
            "poly_query_s": 0.0,
            "dist_s": 0.0,
            "write_s": 0.0,
        }

        N = len(lon_chunk)
        dist = np.empty(N, dtype=np.float32)
        c1   = np.empty(N, dtype=np.int32)
        c2   = np.empty(N, dtype=np.int32)

        mb = (is_border_chunk == 1)
        # ---------- Near-border fast path ----------
        if np.any(mb):
            idxs = np.where(mb)[0]
            stats["fast"] += int(idxs.size)
            dist[mb] = dkm_hint_chunk[mb]  # trust offset magnitude
            for i in idxs:
                lon = float(lon_chunk[i]); lat = float(lat_chunk[i])
                # Decide side via prepared covers on the two known neighbors
                t_poly0 = time.perf_counter()
                ida = int(ida_chunk[i]); idb = int(idb_chunk[i])
                pgeom_a = _BS._id2geom_prepared.get(ida)
                pgeom_b = _BS._id2geom_prepared.get(idb)
                pt = Point(lon, lat)
                decided = False
                if pgeom_a is not None and pgeom_a.covers(pt):
                    c1[i], c2[i] = ida, idb
                    decided = True
                elif pgeom_b is not None and pgeom_b.covers(pt):
                    c1[i], c2[i] = idb, ida
                    decided = True
                t_poly1 = time.perf_counter()
                stats["poly_query_s"] += (t_poly1 - t_poly0)

                if not decided:
                    # fallback to full nearest (rare)
                    t_d0 = time.perf_counter()
                    d_km, a, b = _BS._nearest_segment_vectorized(lon, lat)
                    t_d1 = time.perf_counter()
                    stats["dist_s"] += (t_d1 - t_d0)
                    dist[i] = d_km
                    c1[i], c2[i] = (a, b)

        # ---------- Uniform full path ----------
        mu = ~mb
        if np.any(mu):
            idxs = np.where(mu)[0]
            stats["full"] += int(idxs.size)
            for i in idxs:
                lon = float(lon_chunk[i]); lat = float(lat_chunk[i])
                t_d0 = time.perf_counter()
                d_km, a, b = _BS._nearest_segment_vectorized(lon, lat)
                t_d1 = time.perf_counter()
                stats["dist_s"] += (t_d1 - t_d0)

                dist[i] = d_km
                c1_id = _BS._country_id_at_lonlat(lon, lat)
                if c1_id is None:
                    c1[i], c2[i] = (a, b)
                else:
                    c1[i] = int(c1_id)
                    c2[i] = b if c1[i] == a else a

        # ---------- write ----------
        t_w0 = time.perf_counter()
        df = pd.DataFrame({
            "lon": lon_chunk.astype(np.float32),
            "lat": lat_chunk.astype(np.float32),
            "x": xyz_chunk[:,0].astype(np.float32),
            "y": xyz_chunk[:,1].astype(np.float32),
            "z": xyz_chunk[:,2].astype(np.float32),
            "dist_km": dist,
            "c1_id": c1,
            "c2_id": c2,
            "is_border": is_border_chunk.astype(np.uint8),
            "r_band": r_band_chunk.astype(np.uint8),
        })
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, temp_path, compression="zstd")
        t_w1 = time.perf_counter()
        stats["write_s"] += (t_w1 - t_w0)

        # ---------- done ----------
        elapsed = time.perf_counter() - t0
        print(f"Chunk {chunk_idx} of size {N} finished by process {_PID} aka {_PROCNAME}: {elapsed:.3f}s")
        return ("ok", {
            "path": temp_path,
            "pid": _PID, "proc": _PROCNAME,
            "elapsed": elapsed, "rows": int(N),
            "stats": stats
        })

    except Exception as e:
        import traceback
        return ("err", {"error": str(e), "traceback": traceback.format_exc()})

# --------------------------- concat utility -------------------------

def _concat_parquet_shards(shard_paths: list[str], out_path: str,
                           compression: str = "zstd", row_group_size: int | None = 512_000):
    """
    Concatenate many shards into **one Parquet file** by streaming row-groups.
    Stable across PyArrow versions, memory-bounded, and fast.
    """
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
            del pf  # important on Windows — release file handle
    finally:
        if writer is not None:
            writer.close()

# --------------------------- public API -----------------------------

def make_dataset_parallel(
    n_total: int,
    out_path: str = os.path.join(FOLDER_PATH, "dataset_all.parquet"),
    mixture = (0.70, 0.30),          # (near-border, uniform)
    shards_per_total: int = 24,      # total temp shards to produce (↑ keeps workers busy)
    max_workers: int | None = None,  # None => CPU-1
    seed: int | None = 42,           # RNG seed for reproducibility (None => nondeterministic)
    gpkg_path: str = GPKG_PATH,
    countries_layer: str = COUNTRIES_LAYER,
    id_field: str = ID_FIELD,
    borders_fgb: str = BORDERS_FGB,
    tmp_subdir: str = "tmp_shards",
    writer_compression: str = "zstd",
    writer_row_group_size: int | None = 512_000,
) -> str:
    """
    Generate n_total samples and write ONE Parquet at `out_path`.

    Returns:
        The absolute path to the written Parquet file.
    """
    if cKDTree is None:
        raise RuntimeError("scipy.spatial.cKDTree is required. `pip install scipy`.")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_dir = os.path.dirname(out_path) or "."
    tmp_dir = os.path.join(out_dir, tmp_subdir)
    os.makedirs(tmp_dir, exist_ok=True)

    # RNG: fixed seed => same dataset every time; None => new randomness
    rng = np.random.default_rng(seed)

    t0 = time.perf_counter()
    # Sampler used **only** to draw near-border points in the parent (no labeling here)
    sampler_for_drawing = BorderSampler(
        gpkg_path=gpkg_path,
        countries_layer=countries_layer,
        id_field=id_field,
        borders_fgb=borders_fgb,
    )
    dt = time.perf_counter() - t0
    print(f"Sampler creation: {dt:.3f}s")
    t0 = time.perf_counter()
    
    # 1) Draw points (parent process, fast)
    n_border = int(mixture[0] * n_total)
    n_uniform = n_total - n_border

    (lon_b, lat_b, xyz_b, is_b_b, r_band_b, dkm_b, ida_b, idb_b) = \
        _draw_near_border_points(sampler_for_drawing, n_border, rng)

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

    # Shuffle (so shards have similar composition)
    perm = rng.permutation(len(lon))
    lon, lat, xyz, is_border, r_band, dkm_hint, ida, idb = \
        lon[perm], lat[perm], xyz[perm], is_border[perm], r_band[perm], dkm_hint[perm], ida[perm], idb[perm]

    dt = time.perf_counter() - t0
    print(f"Parent Processes: {dt:.3f}s")
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
    agg = defaultdict(float)   # holds sums
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(gpkg_path, countries_layer, id_field, borders_fgb),
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
                tmp_path, ci
            )
            futures.append(ex.submit(_label_and_write_chunk, args))

        it = as_completed(futures)
        if tqdm: it = tqdm(it, total=len(futures), desc="Labeling shards")
        for f in it:
            status, payload = f.result()
            if status == "err":
                raise RuntimeError(f"Worker failed: {payload['error']}\n{payload['traceback']}")
            info = payload
            temp_paths.append(info["path"]) 

            # aggregate stats
            s = info.get("stats", {})
            agg["points"]        += s.get("points", 0)
            agg["fast"]          += s.get("fast", 0)
            agg["full"]          += s.get("full", 0)
            agg["poly_query_s"]  += s.get("poly_query_s", 0.0)
            agg["dist_s"]        += s.get("dist_s", 0.0)
            agg["write_s"]       += s.get("write_s", 0.0)

    # after the loop, emit a compact summary
    points = int(agg["points"])
    fast = int(agg["fast"])
    full = int(agg["full"])
    print(
        "[aggregate] "
        f"points={points} fast={fast} full={full}  "
        f"t_poly={agg['poly_query_s']:.1f}s t_dist={agg['dist_s']:.1f}s t_write={agg['write_s']:.1f}s"
    )

    dt = time.perf_counter() - t0
    print(f"Parallel labeling: {dt:.3f}s")
    t0 = time.perf_counter()
    # 3) Concatenate shards -> ONE Parquet
    _concat_parquet_shards(temp_paths, out_path, compression=writer_compression,
                           row_group_size=writer_row_group_size)

    # 4) Cleanup temporary files
    for p in temp_paths:
        try: os.remove(p)
        except Exception: pass
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass
    
    dt = time.perf_counter() - t0
    print(f"Concatenation and Clean up: {dt:.3f}s")

    return os.path.abspath(out_path)

# --------------------------- script example -------------------------

if __name__ == "__main__":
    # safety for multiprocessing
    mp.set_start_method("spawn", force=True)

    t0 = time.perf_counter()
    path = make_dataset_parallel(
        n_total=32_000,
        out_path=os.path.join(FOLDER_PATH, "dataset_all.parquet"),
        mixture=(0.70, 0.30),
        shards_per_total=32,
        max_workers=None,   
        seed=37,           
    )
    dt = time.perf_counter() - t0
    print(f"Total time Elapsed: {dt:.3f}s")
