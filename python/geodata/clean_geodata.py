"""
geodata_dataset.py
==================

Minimal, production-ready generator for (lon, lat, x, y, z, dist_km, c1_id, c2_id, is_border, r_band)
saved as **one Parquet file**.

Design choices (why this looks the way it does):
- We **bias sampling near borders** (70/30 by default) because the regression/classification tasks
  are hardest there. Uniform samples still cover the globe for stability.
- Border distances are computed **on the sphere** (short-arc great-circle segments); coastlines
  are just borders with the water polygon.
- We precompute border segments externally (borders.fgb) and load them once per worker.
- Labeling is parallelized: each worker builds its own spatial indices (GDAL/GEOS init is heavy),
  labels a chunk, writes a **temporary Parquet shard**. The parent then **concatenates shards**
  row-group-by-row-group into **one file**. (No pyarrow.dataset/scanner — those APIs change across versions.)
- Everything is **Windows/macOS-safe** (spawn start method, __main__ guard).
"""

from __future__ import annotations
from utils import _save_points_parquet, _load_points_parquet

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
from shapely.strtree import STRtree
try:
    # optional progress bar; gracefully degrades to range()
    from tqdm import tqdm
except Exception:
    tqdm = None

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
    Unsigned spherical distance (km) from P to the **short** arc A-B.
    Handles degenerate segments by falling back to endpoint distance.
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

# --------------------------- labeling core --------------------------

class BorderSampler:
    """
    Minimal spatial labeller:
    - countries: polygon coverage test -> c1_id
    - borders:   pre-segmented line segments (lon/lat); spherical P→segment distance
    Why STRtrees? Because country coverage and nearest-segment shortlist need fast spatial filtering.
    """

    def __init__(
        self,
        gpkg_path: str = GPKG_PATH,
        countries_layer: str = COUNTRIES_LAYER,
        id_field: str = ID_FIELD,
        borders_fgb: str = BORDERS_FGB,
    ):
        # Countries (WGS84 lon/lat)
        cdf = gpd.read_file(gpkg_path, layer=countries_layer)
        if cdf.crs is None or (cdf.crs.to_epsg() or 4326) != 4326:
            cdf = cdf.to_crs(4326)
        self.id_field = id_field
        self.countries = cdf[[id_field, "geometry"]].reset_index(drop=True)

        self._poly_geoms = list(self.countries.geometry.values)
        self._poly_tree  = STRtree(self._poly_geoms)
        self._poly_wkb2pos = {g.wkb: i for i, g in enumerate(self._poly_geoms)}

        # Borders (FlatGeobuf with segmentized shared boundaries)
        bdf = gpd.read_file(borders_fgb)
        if bdf.crs is None or (bdf.crs.to_epsg() or 4326) != 4326:
            bdf = bdf.to_crs(4326)
        self.borders = bdf.reset_index(drop=True)

        self._seg_geoms = list(self.borders.geometry.values)
        self._seg_tree  = STRtree(self._seg_geoms)
        self._seg_wkb2pos = {g.wkb: i for i, g in enumerate(self._seg_geoms)}

        # Fast access arrays for distance loop
        self._ax = self.borders["ax"].to_numpy()
        self._ay = self.borders["ay"].to_numpy()
        self._bx = self.borders["bx"].to_numpy()
        self._by = self.borders["by"].to_numpy()
        self._id_a = self.borders["id_a"].to_numpy(dtype=int)
        self._id_b = self.borders["id_b"].to_numpy(dtype=int)

        # === Precompute segment geometry for fast vectorized distance ===
        # Unit vectors for endpoints
        A3 = lonlat_to_unitvec(self._ax, self._ay).astype(np.float64)   # (S,3)
        B3 = lonlat_to_unitvec(self._bx, self._by).astype(np.float64)   # (S,3)

        # Per-segment normal to great circle (unit)
        N3 = np.cross(A3, B3)
        N3 = _safe_div(N3, _safe_norm(N3))

        # cos and theta of the short arc A->B
        cos_ab = np.clip((A3 * B3).sum(axis=1), -1.0, 1.0)
        theta_ab = np.arccos(cos_ab)

        # Store for workers (float64 for numeric stability)
        self._A3 = A3
        self._B3 = B3
        self._N3 = N3
        self._cos_ab = cos_ab
        self._theta_ab = theta_ab

        
    # ---- small helpers

    def _to_positions(self, res, wkb2pos):
        """STRtree returns either ints or geometries depending on Shapely version."""
        arr = np.asarray(res)
        if arr.dtype.kind in ("i", "u"):  # already positions
            return arr
        return np.array([wkb2pos[g.wkb] for g in res], dtype=int)

    def _country_id_at_lonlat(self, lon, lat) -> int | None:
        """Find c1_id by coverage; `covers` handles on-border cases gracefully."""
        pt = Point(lon, lat)
        pos = self._to_positions(self._poly_tree.query(pt), self._poly_wkb2pos)
        for i in pos:
            if self._poly_geoms[i].covers(pt):
                return int(self.countries.iloc[i][self.id_field])
        return None

    def _candidate_segment_indices(self, pt: Point) -> np.ndarray:
        """
        Shortlist candidate border segments near point.
        Strategy: nearest() if available, then grow radius until we catch neighbors.
        Fallback: fixed radii. This keeps the shortlist small (fast distance loop) and robust.
        """
        try:
            nearest_geom = self._seg_tree.nearest(pt)
            nearest_idx  = self._seg_wkb2pos[nearest_geom.wkb]
            d0 = pt.distance(nearest_geom)
            base = max(d0 * 1.5, 0.05)
            r = base
            while r <= 30.0:
                cand = self._to_positions(self._seg_tree.query(pt.buffer(r)), self._seg_wkb2pos)
                if len(cand) > 0:
                    return cand
                r *= 2.0
            return np.array([nearest_idx], dtype=int)
        except Exception:
            pass

        for r in (0.25, 1.0, 3.0, 7.0, 15.0, 30.0):
            cand = self._to_positions(self._seg_tree.query(pt.buffer(r)), self._seg_wkb2pos)
            if len(cand) > 0:
                return cand
        return np.array([], dtype=int)

    # ---- public labeling

    def sample_lonlat(self, lon: float, lat: float) -> tuple[float, int, int]:
        """
        Return (distance_km [unsigned], c1_id, c2_id) at (lon,lat).
        c2_id is the *other* country of the nearest border segment.
        """
        pt = Point(lon, lat)

        c1 = self._country_id_at_lonlat(lon, lat)

        cand_idx = self._candidate_segment_indices(pt)
        if len(cand_idx) == 0:
            return float("nan"), int(c1) if c1 is not None else -1, -1

        # Evaluate spherical distance on the shortlist
        best_d = 1e18
        best_pair = (None, None)
        ax = self._ax[cand_idx]; ay = self._ay[cand_idx]
        bx = self._bx[cand_idx]; by = self._by[cand_idx]
        ida = self._id_a[cand_idx]; idb = self._id_b[cand_idx]
        for i in range(len(cand_idx)):
            d = greatcircle_point_segment_dist_km(lon, lat, ax[i], ay[i], bx[i], by[i])
            if d < best_d:
                best_d = d
                best_pair = (int(ida[i]), int(idb[i]))

        a, b = best_pair
        if c1 == a:
            c2 = b
        elif c1 == b:
            c2 = a
        else:
            # Rare tiny-island/topology corner case.
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
_BS = None
_ID2GEOM = None
_PID = None
_PROCNAME = None

def _init_worker(gpkg_path, layer, id_field, borders_fgb):
    """
    Build spatial indices ONCE per worker.
    Why? GEOS/STRtree init is expensive; we want to amortize it across a lot of points.
    """
    global _BS, _ID2GEOM, _PID, _PROCNAME
    _BS = BorderSampler(gpkg_path, layer, id_field, borders_fgb)
    # quick map for 2x 'covers' checks when we already know the two adjacent countries
    _ID2GEOM = {int(row[id_field]): geom for row, geom in
                zip(_BS.countries.to_dict("records"), _BS._poly_geoms)}

    import os, multiprocessing as _mp
    _PID = os.getpid()
    _PROCNAME = _mp.current_process().name
    
def _nearest_segment_vectorized(lon: float, lat: float, cand_idx: np.ndarray):
    """
    Vectorized nearest short-arc distance using precomputed A3, B3, N3.
    Returns (dist_km, id_a, id_b). Assumes cand_idx is non-empty.
    """
    # point as unit vector
    p = lonlat_to_unitvec(np.array([lon], dtype=np.float64),
                          np.array([lat], dtype=np.float64))[0]  # (3,)

    A = _BS._A3[cand_idx]       # (K,3)
    B = _BS._B3[cand_idx]       # (K,3)
    N = _BS._N3[cand_idx]       # (K,3)
    th_ab = _BS._theta_ab[cand_idx]  # (K,)

    # Projection of p to each candidate great circle
    # c_k = normalize( (N_k x p) x N_k )
    C = np.cross(N, p)                 # (K,3)
    C = np.cross(C, N)                 # (K,3)
    C = _safe_div(C, _safe_norm(C))

    # Check if projection lies on the short arc: |AC| + |CB| == |AB|
    # Use dot/cosine forms with clamp for stability
    def arccos_safe(dotv):  # vectorized arccos with clip
        return np.arccos(np.clip(dotv, -1.0, 1.0))

    ac = arccos_safe(np.sum(A * C, axis=1))     # (K,)
    cb = arccos_safe(np.sum(C * B, axis=1))     # (K,)

    on_short = np.abs((ac + cb) - th_ab) < 1e-10

    # Distance via projection (on_short) or min to endpoints (off-short)
    theta_proj = arccos_safe(np.dot(C, p))      # (K,)
    theta_pa   = arccos_safe(np.dot(A, p))
    theta_pb   = arccos_safe(np.dot(B, p))
    theta_off  = np.minimum(theta_pa, theta_pb)

    theta = np.where(on_short, theta_proj, theta_off)  # (K,)
    k = int(np.argmin(theta))
    d_km = float(R_EARTH_KM * theta[k])

    ida = int(_BS._id_a[cand_idx[k]])
    idb = int(_BS._id_b[cand_idx[k]])
    return d_km, ida, idb

    
def _label_and_write_chunk(args):
    """
    Label a chunk and write a TEMP Parquet shard.

    Instrumentation:
      - counts: points, fast (near-border) vs full (uniform) paths
      - timings: polygon covers, segment STRtree query, distance loop, parquet write
      - candidate stats: sum & max length of candidate shortlists (full path + fast fallback)
    """
    import time, os
    t0 = time.perf_counter()
    try:
        (gpkg_path, layer, id_field, borders_fgb,
         lon_chunk, lat_chunk, xyz_chunk, is_border_chunk, r_band_chunk,
         dkm_hint_chunk, ida_chunk, idb_chunk,
         temp_path, chunk_idx) = args

        global _BS, _ID2GEOM, _PID, _PROCNAME
        if _BS is None or _ID2GEOM is None:
            _init_worker(gpkg_path, layer, id_field, borders_fgb)

        # ---------- stats bucket ----------
        stats = {
            "points": int(len(lon_chunk)),
            "fast": 0,
            "full": 0,
            "poly_query_s": 0.0,
            "seg_query_s": 0.0,
            "dist_s": 0.0,
            "write_s": 0.0,
            "cand_count_sum": 0,
            "cand_count_max": 0,
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
                pt = Point(lon, lat)
                ida = int(ida_chunk[i]); idb = int(idb_chunk[i])

                # time polygon coverage checks
                t_poly0 = time.perf_counter()
                ga = _ID2GEOM.get(ida);   gb = _ID2GEOM.get(idb)
                decided = False
                if ga is not None and ga.covers(pt):
                    c1[i], c2[i] = ida, idb
                    decided = True
                elif gb is not None and gb.covers(pt):
                    c1[i], c2[i] = idb, ida
                    decided = True
                t_poly1 = time.perf_counter()
                stats["poly_query_s"] += (t_poly1 - t_poly0)

                if not decided:
                    # fallback: measure seg query + distance loop
                    t_seg0 = time.perf_counter()
                    cand_idx = _BS._candidate_segment_indices(pt)
                    t_seg1 = time.perf_counter()
                    stats["seg_query_s"] += (t_seg1 - t_seg0)

                    cands = int(len(cand_idx))
                    stats["cand_count_sum"] += cands
                    stats["cand_count_max"] = max(stats["cand_count_max"], cands)

                    if cands == 0:
                        # extremely unlikely; keep ids as hint
                        c1[i], c2[i] = ida, idb
                    else:
                        t_d0 = time.perf_counter()
                        best_d = 1e18; best_ids = (None, None)
                        ax = _BS._ax[cand_idx]; ay = _BS._ay[cand_idx]
                        bx = _BS._bx[cand_idx]; by = _BS._by[cand_idx]
                        ida_arr = _BS._id_a[cand_idx]; idb_arr = _BS._id_b[cand_idx]
                        for k in range(cands):
                            d = greatcircle_point_segment_dist_km(lon, lat, ax[k], ay[k], bx[k], by[k])
                            if d < best_d:
                                best_d = d
                                best_ids = (int(ida_arr[k]), int(idb_arr[k]))
                        t_d1 = time.perf_counter()
                        stats["dist_s"] += (t_d1 - t_d0)

                        dist[i] = best_d
                        aa, bb = best_ids
                        # choose c1 by coverage if possible, else use nearest pair
                        if ga is not None and ga.covers(pt):
                            c1[i], c2[i] = ida, idb
                        elif gb is not None and gb.covers(pt):
                            c1[i], c2[i] = idb, ida
                        else:
                            c1[i], c2[i] = aa, bb

        # ---------- Uniform full path ----------
        mu = ~mb
        if np.any(mu):
            idxs = np.where(mu)[0]
            stats["full"] += int(idxs.size)

            for i in idxs:
                lon = float(lon_chunk[i]); lat = float(lat_chunk[i])
                pt = Point(lon, lat)

                # measure seg query
                t_seg0 = time.perf_counter()
                cand_idx = _BS._candidate_segment_indices(pt)
                t_seg1 = time.perf_counter()
                stats["seg_query_s"] += (t_seg1 - t_seg0)

                cands = int(len(cand_idx))
                stats["cand_count_sum"] += cands
                stats["cand_count_max"] = max(stats["cand_count_max"], cands)

                if cands == 0:
                    dist[i], c1[i], c2[i] = float("nan"), -1, -1
                    continue

                # measure distance loop
                t_d0 = time.perf_counter()
                best_d = 1e18; best_ids = (None, None)
                ax = _BS._ax[cand_idx]; ay = _BS._ay[cand_idx]
                bx = _BS._bx[cand_idx]; by = _BS._by[cand_idx]
                ida_arr = _BS._id_a[cand_idx]; idb_arr = _BS._id_b[cand_idx]
                for k in range(cands):
                    d = greatcircle_point_segment_dist_km(lon, lat, ax[k], ay[k], bx[k], by[k])
                    if d < best_d:
                        best_d = d
                        best_ids = (int(ida_arr[k]), int(idb_arr[k]))
                t_d1 = time.perf_counter()
                stats["dist_s"] += (t_d1 - t_d0)

                dist[i] = best_d
                aa, bb = best_ids

                # containing country
                c1_id = _BS._country_id_at_lonlat(lon, lat)
                if c1_id is None:
                    c1[i], c2[i] = aa, bb
                else:
                    c1[i] = int(c1_id)
                    c2[i] = bb if c1[i] == aa else aa

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


def _concat_parquet_shards(shard_paths: list[str], out_path: str,
                           compression: str = "zstd", row_group_size: int | None = 512_000):
    """
    Concatenate many shards into **one Parquet file** by streaming row-groups.
    Why this way? It's stable across PyArrow versions, memory-bounded, and fast.
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
                # Arrow will split into RGs if row_group_size is provided
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
    points_path: str | None = None,         # if given, load starting points from here
    export_points_path: str | None = None,  # if given (and points_path is None), save the sampled points here
    shuffle_points: bool = True,            # set False to keep exact order when exporting/consuming
) -> str:
    """
    Generate n_total samples and write ONE Parquet at `out_path`.

    Returns:
        The absolute path to the written Parquet file.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_dir = os.path.dirname(out_path) or "."
    tmp_dir = os.path.join(out_dir, tmp_subdir)
    os.makedirs(tmp_dir, exist_ok=True)

    # RNG: fixed seed => same dataset every time; None => new randomness
    rng = np.random.default_rng(seed)
    
    
    # 1) Draw points (parent process, fast)
    # Obtain starting points (either load or generate)
    if points_path is not None:
        # --- apples-to-apples path: load pre-generated points and skip any RNG usage here ---
        t0 = time.perf_counter()
        (lon, lat, xyz, is_border, r_band, dkm_hint, ida, idb) = _load_points_parquet(points_path)
        dt = time.perf_counter() - t0
        print(f"Loaded starting points from '{points_path}' in {dt:.3f}s")
        # honor user choice about shuffling (default True to mix shards; False for strictly identical order)
        if shuffle_points:
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(lon))
            lon, lat, xyz, is_border, r_band, dkm_hint, ida, idb = \
                lon[perm], lat[perm], xyz[perm], is_border[perm], r_band[perm], dkm_hint[perm], ida[perm], idb[perm]
    else:
        # --- normal path: generate points here, optionally export for reuse by other labelers ---
        # Sampler used **only** to draw near-border points in the parent (no labeling here)
        t0 = time.perf_counter()
        sampler_for_drawing = BorderSampler(
            gpkg_path=gpkg_path,
            countries_layer=countries_layer,
            id_field=id_field,
            borders_fgb=borders_fgb,
        )
        dt = time.perf_counter() - t0
        print(f"Sampler creation: {dt:.3f}s")
        t0 = time.perf_counter()

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

        if shuffle_points:
            perm = rng.permutation(len(lon))
            lon, lat, xyz, is_border, r_band, dkm_hint, ida, idb = \
                lon[perm], lat[perm], xyz[perm], is_border[perm], r_band[perm], dkm_hint[perm], ida[perm], idb[perm]

        if export_points_path:
            ep = _save_points_parquet(export_points_path, lon, lat, xyz, is_border, r_band, dkm_hint, ida, idb)
            print(f"Exported starting points to '{ep}'")


    dt = time.perf_counter() - t0
    print(f"Parent Processes: {dt:.3f}s")
    t0 = time.perf_counter()
    
    # 2) Parallel labeling into many temp shards
    total = len(lon)
    shards = max(1, int(shards_per_total))
    # Keep shard size decently large to amortize per-task overhead
    chunk_size = max(1, int(np.ceil(total / shards)))

    if max_workers is None:
        try:
            max_workers = max(1, (os.cpu_count() or 4) - 1)
        except Exception:
            max_workers = 4

    futures, temp_paths = [], []
    agg = defaultdict(float)   # holds sums and max
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
            agg["seg_query_s"]   += s.get("seg_query_s", 0.0)
            agg["dist_s"]        += s.get("dist_s", 0.0)
            agg["write_s"]       += s.get("write_s", 0.0)
            agg["cand_count_sum"]+= s.get("cand_count_sum", 0)
            # track maximum
            agg["cand_count_max"] = max(agg.get("cand_count_max", 0), s.get("cand_count_max", 0))

    # after the loop, emit a compact summary
    points = int(agg["points"])
    fast = int(agg["fast"])
    full = int(agg["full"])
    avg_cands = (agg["cand_count_sum"] / max(1, full + (fast)))  # includes fast fallbacks
    print(
        "[aggregate] "
        f"points={points} fast={fast} full={full}  "
        f"t_poly={agg['poly_query_s']:.1f}s t_seg={agg['seg_query_s']:.1f}s t_dist={agg['dist_s']:.1f}s t_write={agg['write_s']:.1f}s  "
        f"avg_cands≈{avg_cands:.1f} max_cands={int(agg['cand_count_max'])}"
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
    t0 = time.perf_counter()

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
    Usage (CLI): python clean_geodata.py --lon -56.32122 --lat 47.07616
    """
    bs = BorderSampler(
        gpkg_path=gpkg_path,
        countries_layer=countries_layer,
        id_field=id_field,
        borders_fgb=borders_fgb,
    )
    d_km, c1, c2 = bs.sample_lonlat(float(lon), float(lat))
    print(f"== lon: {float(lon):.6f}, lat: {float(lat):.6f}, dist_km: {d_km:.6f}, c1: {c1}, c2: {c2} ==")

# --------------------------- script example -------------------------

def run():
    # safety for multiprocessing
    mp.set_start_method("spawn", force=True)

    t0 = time.perf_counter()
    path = make_dataset_parallel(
        n_total=32_000,
        out_path=os.path.join(FOLDER_PATH, "dataset_all.parquet"),
        mixture=(0.70, 0.30),
        shards_per_total=32,
        max_workers=None,   
        seed=None,           
    )
    dt = time.perf_counter() - t0
    print(f"Total time Elapsed: {dt:.3f}s")

if __name__ == "__main__":
    run()
    #query_point(-56.32122, 47.07616)  # Newfoundland, Canada
