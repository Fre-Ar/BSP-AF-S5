# src/geodata/sampler/labeling.py

# --- must be first import
from __future__ import annotations


# --- stdlib
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
from typing import Callable

# --- third-party
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry import Point

try:
    from tqdm import tqdm  # optional progress bar
except Exception:
    tqdm = None
    
# --- our modules
from utils.utils import (
    _concat_parquet_shards
)
from utils.utils_geo import (
    GPKG_PATH,
    BORDERS_FGB_PATH,
    COUNTRIES_LAYER,
    ID_FIELD,
    FOLDER_PATH,
    SEED,
    unitvec_to_lonlat,
    normalize_vec
)
from .sampling import BorderSampler, _draw_near_border_points



# --------------------------- CONFIG ---------------------------
    
# When True, near-border hints are validated once via a full KNN lookup.
VALIDATE_NEAR_BORDER = True
# If d_knn < HINT_DELTA_FACTOR * d_hint, we override the hint distance.
HINT_DELTA_FACTOR = 0.8

# --------------------------- multiprocessing ------------------------

# Globals living inside workers (one-time init per process)
_BS: BorderSampler | None = None
_PID = None
_PROCNAME = None

def _init_worker(
    gpkg_path: str,
    layer: str,
    id_field: str,
    borders_fgb: str,
    knn_k: int,
    knn_expand: int,
    expand_rel: float,
):
    """
    Worker initializer.

    Builds a BorderSampler once per process, so each worker has its own
    in-memory spatial structures (GEOS, KDTree, etc.).
    """
    global _BS, _PID, _PROCNAME
    _BS = BorderSampler(
        gpkg_path=gpkg_path,
        countries_layer=layer,
        id_field=id_field,
        borders_fgb=borders_fgb,
        knn_k=knn_k,
        knn_expand=knn_expand,
        expand_rel=expand_rel,
    )
    import multiprocessing as _mp
    _PID = os.getpid()
    _PROCNAME = _mp.current_process().name
    
    
def _label_rows(
    mode: str,
    lon: np.ndarray,
    lat: np.ndarray,
    is_border: np.ndarray,
    r_band: np.ndarray,
    dkm_hint: np.ndarray,
    ida: np.ndarray,
    idb: np.ndarray,
    stats: dict,
):
    """
    Labels a batch of points with distances and (c1_id, c2_id).

    Parameters
    ----------
    mode : {"hybrid", "reliable"}
        - "hybrid": trust near-border hints for is_border==1 points, validating once
          via KNN; use full KNN for others.
        - "reliable": ignore hints; use the full KNN/country lookup for all rows.
    lon, lat : ndarray
        Arrays of longitudes and latitudes.
    is_border : ndarray
        1 for points sampled via the near-border process, else 0.
    r_band : ndarray
        Near-border radial band indices.
    dkm_hint : ndarray
        Distance hints produced by near-border sampling (km). May be 0 for uniform samples.
    ida, idb : ndarray
        Country IDs on either side of the border segment used when sampling near-border
        points. For uniform points these will be zeros.
    stats : dict
        Mutable dict to accumulate timing/counter statistics.

    Returns
    -------
    dist : ndarray
        Distance to nearest border (km) for each point.
    c1 : ndarray
        Containing country ID for each point (from polygon coverage when available).
    c2 : ndarray
        Adjacent country ID on the nearest border.
    N : int
        Number of points processed.
    """
    assert _BS is not None, "BorderSampler must be initialized in worker via _init_worker()."
    
    N = len(lon)
    dist = np.empty(N, np.float32)
    c1 = np.empty(N, np.int32)
    c2 = np.empty(N, np.int32)
    mb = (is_border == 1)

    def nearest(lon_i, lat_i):
        """
        Helper: nearest border distance + border pair via KDTree + spherical kernel.
        """
        t0 = time.perf_counter()
        d_km, a, b = _BS._nearest_segment_vectorized(lon_i, lat_i)
        stats["dist_s"] += time.perf_counter() - t0
        return d_km, a, b

    # ---------------------------------------------------------
    # 1) Near-border fast path (only when mode == "hybrid")
    #    We trust the offset magnitude (dkm_hint) and try to decide the side
    #    using prepared polygon covers, then optionally validate.
    # ---------------------------------------------------------
    if mode == "hybrid" and np.any(mb):
        idxs = np.where(mb)[0]
        stats["fast"] += int(idxs.size)
        dist[mb] = dkm_hint[mb]
        for i in idxs:
            # get point
            lon_i = float(lon[i])
            lat_i = float(lat[i])
            pt = Point(lon_i, lat_i)
            
            # get prepared ids and prepared geometries
            ida_i = int(ida[i])
            idb_i = int(idb[i])
            pa = _BS._id2geom_prepared.get(ida_i)
            pb = _BS._id2geom_prepared.get(idb_i)
            
            # checking which of the prepared ids lies on the side of the point
            decided = False
            t0p = time.perf_counter()
            # Decide c1/c2 purely from which country polygon covers the point
            if pa is not None and pa.covers(pt):
                c1[i], c2[i] = ida_i, idb_i
                decided = True
            elif pb is not None and pb.covers(pt): 
                c1[i], c2[i] = idb_i, ida_i
                decided = True
            stats["poly_query_s"] += time.perf_counter() - t0p
            
            # Default: assume we will keep the hint unless validation disagrees
            use_hint = True 
            
            if VALIDATE_NEAR_BORDER:
                # One single KNN nearest-segment check
                d_knn, a_knn, b_knn = nearest(lon_i, lat_i)
                
                same_pair = (
                    (a_knn == ida_i and b_knn == idb_i)
                    or (a_knn == idb_i and b_knn == ida_i)
                )
                
                # If nearest border pair doesn't match the hinted pair, or if we find
                # a border significantly closer than the hint, we override.
                if (not same_pair) or (d_knn < HINT_DELTA_FACTOR * float(dkm_hint[i])):
                    dist[i] = d_knn
                    # If polygon side is unknown or inconsistent, go with nearest pair.
                    if not decided: c1[i], c2[i] = a_knn, b_knn
                    use_hint = False
                elif not decided:
                    # Hinted pair is plausible but we don't know which side:
                    # try polygon coverage again on the KNN pair; fall back to pair order.
                    if _BS._id2geom_prepared.get(a_knn) and _BS._id2geom_prepared[a_knn].covers(pt):
                        c1[i], c2[i] = a_knn, b_knn
                    elif _BS._id2geom_prepared.get(b_knn) and _BS._id2geom_prepared[b_knn].covers(pt):
                        c1[i], c2[i] = b_knn, a_knn
                    else:
                        c1[i], c2[i] = a_knn, b_knn
            
            if use_hint:
                dist[i] = float(dkm_hint[i])

    # ---------------------------------------------------------
    # 2) Full path:
    #    - For uniform points (and all points in "reliable" mode)
    #      we ignore hints and always do a KNN search + polygon coverage.
    # ---------------------------------------------------------
    mu = ~mb if mode == "hybrid" else np.ones(N, dtype=bool)
    if np.any(mu):
        idxs = np.where(mu)[0]
        stats["full"] += int(idxs.size)
        
        for i in idxs:
            lon_i = float(lon[i])
            lat_i = float(lat[i])
            
            # nearest border distance + border pair (a,b)
            d_km, a, b = nearest(lon_i, lat_i)
            dist[i] = d_km
            
            # Containing country via polygon coverage
            c1_id = _BS._country_id_at_lonlat(lon_i, lat_i)
            if c1_id is None:
                c1[i], c2[i] = a, b
            else:
                c1[i] = int(c1_id)
                c2[i] = b if c1[i] == a else a
    return dist, c1, c2, N
    
def _label_and_write_chunk(args):
    """
    Worker entry-point: labels a chunk of points and write a temporary Parquet shard.
    Stats: counts (fast/full), polygon covers time, distance time, write time.

    Parameters
    ----------
    args : tuple
        Packed arguments:
        (gpkg_path, layer, id_field, borders_fgb,
         lon_chunk, lat_chunk, xyz_chunk, is_border_chunk, r_band_chunk,
         dkm_hint_chunk, ida_chunk, idb_chunk,
         temp_path, chunk_idx, knn_k, knn_expand, expand_rel)

    Returns
    -------
    ("ok", payload) or ("err", payload)
        payload for "ok" contains:
            - "path"   : path to shard
            - "pid"    : worker PID
            - "proc"   : worker process name
            - "elapsed": seconds
            - "rows"   : number of rows
            - "stats"  : dict with timing/counter statistics
        payload for "err" contains:
            - "error"    : str message
            - "traceback": formatted traceback
    """
    import time
    t0 = time.perf_counter()
    try:
        # unpack the arguments
        (gpkg_path, layer, id_field, borders_fgb,
         lon_chunk, lat_chunk, xyz_chunk, is_border_chunk, r_band_chunk,
         dkm_hint_chunk, ida_chunk, idb_chunk,
         temp_path, chunk_idx, knn_k, knn_expand, expand_rel) = args

        # init the worker
        global _BS, _PID, _PROCNAME
        if _BS is None:
            _init_worker(gpkg_path, layer, id_field, borders_fgb, knn_k, knn_expand, expand_rel)

        # init stats
        stats = {
            "points": int(len(lon_chunk)),
            "fast": 0,
            "full": 0,
            "poly_query_s": 0.0,
            "dist_s": 0.0,
            "write_s": 0.0,
        }
        
        # label rows
        dist, c1, c2, N = _label_rows(
            mode="reliable",
            lon=lon_chunk,
            lat=lat_chunk,
            is_border=is_border_chunk,
            r_band=r_band_chunk,
            dkm_hint=dkm_hint_chunk,
            ida=ida_chunk,
            idb=idb_chunk,
            stats=stats,
        )

        # Write labeled shard as a parquet file
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
        print(
            f"Chunk {chunk_idx} of size {N} finished by process "
            f"{_PID} aka {_PROCNAME}: {elapsed:.3f}s"
        )
        return "ok", {
            "path": temp_path,
            "pid": _PID,
            "proc": _PROCNAME,
            "elapsed": elapsed,
            "rows": int(N),
            "stats": stats,
        }
    except Exception as e:
        import traceback

        return ("err", {"error": str(e), "traceback": traceback.format_exc()})

# --------------------------- public API -----------------------------

def make_batch_datasets(
    # main args
    n_total_per_file: int,
    num_files: int,
    path_generator: Callable[[int], str],
    mixture = (0.0, 1.0), # (near-border, uniform)
    # multiprocessing knobs
    shards_per_file: int = 24,
    max_workers: int | None = None,
    # sampling args
    seed: int | None = SEED,
    gpkg_path: str = GPKG_PATH,
    countries_layer: str = COUNTRIES_LAYER,
    id_field: str = ID_FIELD,
    borders_fgb: str = BORDERS_FGB_PATH,
    shuffle_points: bool = True,
    # I/O knobs
    tmp_subdir: str = "tmp_shards",
    writer_row_group_size: int | None = 512_000,
    # KNN knobs (propagate to workers)
    knn_k: int = 128,
    knn_expand: int = 256,
    expand_rel: float = 1.05,
) -> list[str]:
    """
    Generates multiple labeled border datasets with n_total_per_file samples in parallel and writes 
    them to multiple Parquet files.
    Reuses the same worker pool and sampler for efficiency.

    Parameters
    ----------
    n_total_per_file : int
        Number of samples per output file.
    num_files : int
        Number of separate files to generate.
    path_generator : callable
        Function taking (file_index) and returning the full output path string.
    mixture : tuple
        (near_border_fraction, uniform_fraction). Must sum to ~1.
    shards_per_file : int
        Approximate number of shards to split `n_total_per_file` into.
    max_workers : int or None
        Max worker processes for ProcessPoolExecutor. If None, uses (cpu_count - 1).
    seed : int or None
        RNG seed for reproducible sampling / shuffling.
    gpkg_path, countries_layer, id_field, borders_fgb : str
        Geodata file and layer names for countries and borders.
    tmp_subdir : str
        Temporary subdirectory under `out_path` directory for shard files.
    writer_compression : str
        Parquet compression codec for the final merged file.
    writer_row_group_size : int or None
        Row group size for Parquet writer; affects compressibility and IO.
    shuffle_points : bool
        If True, randomly permute the sample order before sharding.
    knn_k, knn_expand, expand_rel : numeric
        KNN shortlist parameters passed to BorderSampler.
        
    Returns
    -------
    list[str]
        List of absolute paths to the generated files.
    """
    
    generated_files = []
    
    # Ensure temporary directory exists (we reuse it for all files)
    # We will use the directory of the first file to place the temp folder
    first_path = path_generator(0)
    out_dir_root = os.path.dirname(first_path) or "."
    tmp_dir = os.path.join(out_dir_root, tmp_subdir)
    os.makedirs(tmp_dir, exist_ok=True)
    
    rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # 1) Create shared sampler (parent process only; no labeling here)
    # -----------------------------------------------------
    t0_main = time.perf_counter()
    print("Initializing parent BorderSampler...")
    # create sampler
    sampler_for_drawing = BorderSampler(
        gpkg_path       = gpkg_path,
        countries_layer = countries_layer,
        id_field        = id_field,
        borders_fgb     = borders_fgb,
        knn_k           = knn_k,
        knn_expand      = knn_expand,
        expand_rel      = expand_rel,
    )
    print(f"Sampler creation done in {time.perf_counter() - t0_main:.3f}s. Starting batch generation of {num_files} files.")

    # Determine workers
    if max_workers is None:
        try:
            max_workers = max(1, (os.cpu_count() or 4) - 1)
        except Exception:
            max_workers = 4

    # -----------------------------------------------------
    # 2) Parallel labeling into temporary Parquet shards
    # -----------------------------------------------------
    # Spin up the ProcessPoolExecutor ONCE
    # Workers will init their own BorderSampler once and reuse it for all files.
    with ProcessPoolExecutor(
        max_workers = max_workers,
        mp_context  = mp.get_context("spawn"),
        initializer = _init_worker,
        initargs    = (gpkg_path, countries_layer, id_field, borders_fgb, knn_k, knn_expand, expand_rel),
    ) as ex:
        
        for file_idx in range(num_files):
            t0_file = time.perf_counter()
            out_path = path_generator(file_idx)
            
            # Ensure output dir exists
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            
            print(f"--- Generating File {file_idx+1}/{num_files}: {os.path.basename(out_path)} ---")

            # A) Sample `n_border` points close to the borders
            n_border = int(mixture[0] * n_total_per_file)
            n_uniform = n_total_per_file - n_border

            # near-border
            (lon_b, lat_b, xyz_b, is_b_b, r_band_b, dkm_b, ida_b, idb_b) = \
                _draw_near_border_points(sampler_for_drawing, n_border, rng)
            
            # B) Sample `n_uniform` points uniformly across the globe
            
            # Sample normal(0,1)^3 and normalize to the sphere
            xyz_u = rng.normal(size=(n_uniform, 3)).astype(np.float64)
            xyz_u = normalize_vec(xyz_u).astype(np.float32)
            lon_u, lat_u = unitvec_to_lonlat(xyz_u)
            is_b_u = np.zeros(n_uniform, np.uint8)
            r_band_u = np.full(n_uniform, 255, np.uint8) # 255 marks 'uniform' bucket
            dkm_u = np.zeros(n_uniform, np.float32)
            ida_u = np.zeros(n_uniform, np.int32)
            idb_u = np.zeros(n_uniform, np.int32)
            
            # C) Concatenate near-border and uniform samples
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

            # D) Submit chunks to executor
            total_pts = len(lon)
            shards = max(1, int(shards_per_file))
            chunk_size = max(1, int(np.ceil(total_pts / shards)))
            
            futures = []
            current_temp_paths = []
            
            for ci in range(shards):
                # determine low and high bounds for chunk partitioning
                lo = ci * chunk_size
                hi = min((ci + 1) * chunk_size, total_pts)
                if lo >= hi: 
                    continue
                
                # Unique temp path for this chunk of this file
                tmp_path = os.path.join(tmp_dir, f"f{file_idx}_part{ci:05d}.parquet")
                
                args = (
                    gpkg_path, countries_layer, id_field, borders_fgb,
                    lon[lo:hi], lat[lo:hi], xyz[lo:hi],
                    is_border[lo:hi], r_band[lo:hi], dkm_hint[lo:hi],
                    ida[lo:hi], idb[lo:hi],
                    tmp_path, ci, knn_k, knn_expand, expand_rel
                )
                futures.append(ex.submit(_label_and_write_chunk, args))

            # E) Wait for results
            it = as_completed(futures)
            # show progress bar
            if tqdm: 
                it = tqdm(it, total=len(futures), desc="Labeling shards")
            
            for f in it:
                # aggregate worker stats and payloads
                status, payload = f.result()
                if status == "err":
                    raise RuntimeError(f"Worker failed: {payload['error']}\n{payload['traceback']}")
                current_temp_paths.append(payload["path"])
                
            # F) Concatenate shards into a single Parquet file
            _concat_parquet_shards(current_temp_paths, out_path, row_group_size=writer_row_group_size)
            
            # G) Cleanup temporary files
            for p in current_temp_paths:
                try: os.remove(p)
                except Exception: pass
            
            generated_files.append(os.path.abspath(out_path))
            print(f"File {file_idx+1} completed in {time.perf_counter() - t0_file:.2f}s")
            
    # Cleanup temp dir
    try: os.rmdir(tmp_dir)
    except Exception: pass
    
    print(f"Batch generation finished. Total time: {time.perf_counter() - t0_main:.2f}s")
    return generated_files


def make_dataset_parallel(
    # main args
    n_total: int,
    out_path: str = os.path.join(FOLDER_PATH, "dataset_all.parquet"),
    mixture = (0.70, 0.30),          # (near-border, uniform)
    # multiprocessing knobs
    shards_per_total: int = 24,
    max_workers: int | None = None,
    # sampling args
    seed: int | None = SEED,
    gpkg_path: str = GPKG_PATH,
    countries_layer: str = COUNTRIES_LAYER,
    id_field: str = ID_FIELD,
    borders_fgb: str = BORDERS_FGB_PATH,
    shuffle_points: bool = True,            # False -> keep exact order
    # I/O knobs
    tmp_subdir: str = "tmp_shards",
    writer_row_group_size: int | None = 512_000,
    # KNN knobs (propagate to workers)
    knn_k: int = 128,
    knn_expand: int = 256,
    expand_rel: float = 1.05,
) -> str:
    """
    Generates a labeled border dataset with n_total samples in parallel and writes 
    them to **one** Parquet file.

    Parameters
    ----------
    n_total : int
        Total number of samples to generate (near-border + uniform).
    out_path : str
        Path to the final Parquet file.
    mixture : tuple
        (near_border_fraction, uniform_fraction). Must sum to ~1.
    shards_per_total : int
        Approximate number of shards to split `n_total` into.
    max_workers : int or None
        Max worker processes for ProcessPoolExecutor. If None, uses (cpu_count - 1).
    seed : int or None
        RNG seed for reproducible sampling / shuffling.
    gpkg_path, countries_layer, id_field, borders_fgb : str
        Geodata file and layer names for countries and borders.
    tmp_subdir : str
        Temporary subdirectory under `out_path` directory for shard files.
    writer_compression : str
        Parquet compression codec for the final merged file.
    writer_row_group_size : int or None
        Row group size for Parquet writer; affects compressibility and IO.
    shuffle_points : bool
        If True, randomly permute the sample order before sharding.
    knn_k, knn_expand, expand_rel : numeric
        KNN shortlist parameters passed to BorderSampler.
        
    Returns
    -------
    str
        Absolute path to the written Parquet file.
    """

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_dir = os.path.dirname(out_path) or "."
    tmp_dir = os.path.join(out_dir, tmp_subdir)
    os.makedirs(tmp_dir, exist_ok=True)

    rng = np.random.default_rng(seed)

    # -----------------------------------------------------
    # 1) Draw points (parent process only; no labeling here)
    # -----------------------------------------------------
    t0 = time.perf_counter()
    # create sampler
    sampler_for_drawing = BorderSampler(
        gpkg_path       = gpkg_path,
        countries_layer = countries_layer,
        id_field        = id_field,
        borders_fgb     = borders_fgb,
        knn_k           = knn_k,
        knn_expand      = knn_expand,
        expand_rel      = expand_rel,
    )
    print(f"Sampler creation: {time.perf_counter() - t0:.3f}s")
    t0 = time.perf_counter()

    n_border = int(mixture[0] * n_total)
    n_uniform = n_total - n_border
    
    # 1.1) sample n_border points close to the borders
    (lon_b, lat_b, xyz_b, is_b_b, r_band_b, dkm_b, ida_b, idb_b) = \
        _draw_near_border_points(sampler_for_drawing, n_border, rng)

    # 1.2) sample n_uniform points uniformly across the globe
    
    # sample normal(0,1)^3 and normalize to the sphere
    xyz_u = rng.normal(size=(n_uniform, 3)).astype(np.float64)
    xyz_u = normalize_vec(xyz_u).astype(np.float32)
    lon_u, lat_u = unitvec_to_lonlat(xyz_u)
    is_b_u = np.zeros(n_uniform, np.uint8)
    r_band_u = np.full(n_uniform, 255, np.uint8)  # 255 marks 'uniform' bucket
    dkm_u = np.zeros(n_uniform, np.float32)
    ida_u = np.zeros(n_uniform, np.int32)
    idb_u = np.zeros(n_uniform, np.int32)

    # 1.3) Concatenate near-border and uniform samples
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

    print(f"Parent Processes: {time.perf_counter() - t0:.3f}s")
    t0 = time.perf_counter()
    
    # -----------------------------------------------------
    # 2) Parallel labeling into temporary Parquet shards
    # -----------------------------------------------------
    total = len(lon)
    shards = max(1, int(shards_per_total))
    chunk_size = max(1, int(np.ceil(total / shards)))

    # determine max workers
    if max_workers is None:
        try:
            max_workers = max(1, (os.cpu_count() or 4) - 1)
        except Exception:
            max_workers = 4

    futures, temp_paths = [], []
    agg = defaultdict(float)
    
    # We use spawn start method to be Windows/macOS compatible.
    with ProcessPoolExecutor(
        max_workers = max_workers,
        mp_context  = mp.get_context("spawn"),
        initializer = _init_worker,
        initargs    = (gpkg_path, countries_layer, id_field, borders_fgb, knn_k, knn_expand, expand_rel),
    ) as ex:
        for ci in range(shards):
            # determine low and high bounds for chunk partioning
            lo = ci * chunk_size
            hi = min((ci + 1) * chunk_size, total)
            if lo >= hi:
                continue
            # create worker arguments
            tmp_path = os.path.join(tmp_dir, f"part-{ci:05d}.parquet")
            args = (
                gpkg_path,
                countries_layer,
                id_field,
                borders_fgb,
                lon[lo:hi],
                lat[lo:hi],
                xyz[lo:hi],
                is_border[lo:hi],
                r_band[lo:hi],
                dkm_hint[lo:hi],
                ida[lo:hi],
                idb[lo:hi],
                tmp_path,
                ci,
                knn_k,
                knn_expand,
                expand_rel,
            )
            # call worker
            futures.append(ex.submit(_label_and_write_chunk, args))

        it = as_completed(futures)
        # show progress bar
        if tqdm: 
            it = tqdm(it, total=len(futures), desc="Labeling shards")
        
        for f in it:
            # aggregate worker stats and payloads
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

    # -----------------------------------------------------
    # 3) Concatenate shards into a single Parquet file
    # -----------------------------------------------------
    _concat_parquet_shards(
        temp_paths,
        out_path,
        row_group_size=writer_row_group_size,
    )

    # -----------------------------------------------------
    # 4) Cleanup temporary files
    # -----------------------------------------------------
    for p in temp_paths:
        try: os.remove(p)
        except Exception: pass
    try:
        os.rmdir(tmp_dir)
    except Exception:
        pass
    
    print(f"Concatenation and Clean up: {time.perf_counter() - t0:.3f}s")
    
    # return absolute path
    return os.path.abspath(out_path)

def query_point(
    lon: float,
    lat: float,
    gpkg_path: str = GPKG_PATH,
    countries_layer: str = COUNTRIES_LAYER,
    id_field: str = ID_FIELD,
    borders_fgb: str = BORDERS_FGB_PATH,
):
    """
    Convenience helper: print the labeling for a single (lon, lat) query.

    Parameters
    ----------
    lon, lat : float
        Coordinates in degrees (WGS84).
    gpkg_path, countries_layer, id_field, borders_fgb : str
        Paths/layer names for geodata and borders.
    """
    bs = BorderSampler(
        gpkg_path       = gpkg_path,
        countries_layer = countries_layer,
        id_field        = id_field,
        borders_fgb     = borders_fgb,
    )
    d_km, c1, c2 = bs.sample_lonlat(float(lon), float(lat))
    print(
        f"== lon: {float(lon):.6f}, lat: {float(lat):.6f}, "
        f"dist_km: {d_km:.6f}, c1: {c1}, c2: {c2} =="
    )
