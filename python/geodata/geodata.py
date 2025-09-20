import numpy as np, pandas as pd
import geopandas as gpd
import os, math, itertools, time
import pyarrow as pa, pyarrow.parquet as pq
from shapely.geometry import Point
from shapely.strtree import STRtree
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    import xxhash
except ImportError:
    xxhash = None
try:
    from tqdm import trange
except Exception:
    trange = range
try:
    from tqdm import tqdm
except Exception:
    tqdm = None  # fallback: no progress bar

R_EARTH_KM = 6371.0088  # mean Earth radius
FOLDER_PATH = "python/geodata/"
GPKG_PATH   = "python/geodata/world_bank_geodata.gpkg"
BORDERS_FGB = "python/geodata/borders.fgb"
LAYER       = "countries"
ID_FIELD    = "id"   
 

def xyz_to_lonlat(x, y, z):
    # Assume (x,y,z) is on the unit sphere
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return float(lon), float(lat)

def lonlat_to_unitvec(lon, lat):
    lon = np.radians(lon); lat = np.radians(lat)
    clat = np.cos(lat)
    return np.array([clat*np.cos(lon), clat*np.sin(lon), np.sin(lat)], dtype=float)

def greatcircle_point_segment_dist_km(p_lon, p_lat, a_lon, a_lat, b_lon, b_lat):
    """Unsigned spherical distance (km) from point p to short arc a-b."""
    p = lonlat_to_unitvec(p_lon, p_lat)
    a = lonlat_to_unitvec(a_lon, a_lat)
    b = lonlat_to_unitvec(b_lon, b_lat)

    n = np.cross(a, b); n_norm = np.linalg.norm(n)
    if n_norm == 0.0:
        return R_EARTH_KM * min(
            np.arccos(np.clip(np.dot(p, a), -1.0, 1.0)),
            np.arccos(np.clip(np.dot(p, b), -1.0, 1.0)),
        )
    n /= n_norm

    # foot of perpendicular to GC(a,b)
    c = np.cross(n, p); c = np.cross(c, n); c /= np.linalg.norm(c)

    # arc-membership check for short arc a->b
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
    return R_EARTH_KM * theta

def rand_unit_vec(n: int) -> np.ndarray:
    v = np.random.normal(size=(n, 3)).astype(np.float64)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype(np.float32)

def unitvec_to_lonlat(v: np.ndarray):
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return lon.astype(np.float32), lat.astype(np.float32)

def lonlat_deg_to_unitvec(lon_deg, lat_deg) -> np.ndarray:
    lon = np.radians(lon_deg); lat = np.radians(lat_deg)
    cl = np.cos(lat)
    v = np.stack([cl*np.cos(lon), cl*np.sin(lon), np.sin(lat)], axis=-1)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype(np.float32)

def move_along_geodesic(p: np.ndarray, t: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Move unit vectors p along unit tangents t by angles theta (rad)."""
    ct = np.cos(theta)[:, None]; st = np.sin(theta)[:, None]
    out = p * ct + t * st
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out.astype(np.float32)

def sample_distance_km(batch_size: int):
    """Piecewise log-uniform bands favoring near-border points."""
    # probs over bands: [0–10], [10–50], [50–300] km
    bands = np.random.choice([0, 1, 2], size=batch_size, p=[0.6, 0.3, 0.1])
    low = np.array([0.1, 10.0, 50.0], dtype=np.float32)[bands]  # avoid exactly 0
    high = np.array([10.0, 50.0, 300.0], dtype=np.float32)[bands]
    u = np.random.rand(batch_size).astype(np.float32)
    d = np.exp(np.log(low) + u * (np.log(high) - np.log(low))).astype(np.float32)
    return d, bands.astype(np.uint8)

class BorderSampler:
    """
    sample_xyz(x,y,z) -> (distance_km (positive), country1_id, country2_id)
    sample_lonlat(lon,lat) -> same
    """

    def __init__(self,
                 gpkg_path=GPKG_PATH,
                 countries_layer=LAYER,
                 id_field=ID_FIELD,
                 borders_fgb=BORDERS_FGB):
        # Countries
        cdf = gpd.read_file(gpkg_path, layer=countries_layer)
        if cdf.crs is None or (cdf.crs.to_epsg() or 4326) != 4326:
            cdf = cdf.to_crs(4326)
        self.id_field = id_field
        self.countries = cdf[[id_field, "geometry"]].reset_index(drop=True)
        self._poly_geoms = list(self.countries.geometry.values)
        self._poly_tree = STRtree(self._poly_geoms)
        self._poly_wkb2pos = {g.wkb: i for i, g in enumerate(self._poly_geoms)}

        # Borders
        bdf = gpd.read_file(borders_fgb)
        if bdf.crs is None or (bdf.crs.to_epsg() or 4326) != 4326:
            bdf = bdf.to_crs(4326)
        self.borders = bdf.reset_index(drop=True)
        self._seg_geoms = list(self.borders.geometry.values)
        self._seg_tree = STRtree(self._seg_geoms)
        self._seg_wkb2pos = {g.wkb: i for i, g in enumerate(self._seg_geoms)}

        self._ax = self.borders["ax"].to_numpy()
        self._ay = self.borders["ay"].to_numpy()
        self._bx = self.borders["bx"].to_numpy()
        self._by = self.borders["by"].to_numpy()
        self._id_a = self.borders["id_a"].to_numpy(dtype=int)
        self._id_b = self.borders["id_b"].to_numpy(dtype=int)

    def _to_positions(self, res, wkb2pos):
        arr = np.asarray(res)
        if arr.dtype.kind in ("i", "u"):
            return arr
        return np.array([wkb2pos[g.wkb] for g in res], dtype=int)

    def _country_id_at_lonlat(self, lon, lat):
        pt = Point(lon, lat)
        pos = self._to_positions(self._poly_tree.query(pt), self._poly_wkb2pos)
        for i in pos:
            if self._poly_geoms[i].covers(pt):  # covers handles on-border
                return int(self.countries.iloc[i][self.id_field])
        return None

    def _candidate_segment_indices(self, pt: Point):
        """
        Robust candidate gathering:
        - if STRtree.nearest exists: grow radius from that nearest
        - else: try a sequence of radii
        """
        # Preferred path (Shapely >= 2)
        try:
            nearest_geom = self._seg_tree.nearest(pt)
            nearest_idx = self._seg_wkb2pos[nearest_geom.wkb]
            # Start from a conservative radius; grow until we find something.
            radius_list = []
            d0 = pt.distance(nearest_geom)
            # cap d0 (can be 0 if point is exactly on the segment)
            base = max(d0 * 1.5, 0.05)
            # grow multiplicatively up to 30°
            r = base
            while r <= 30.0:
                radius_list.append(r)
                r *= 2.0
            for r in radius_list:
                cand = self._to_positions(self._seg_tree.query(pt.buffer(r)), self._seg_wkb2pos)
                if len(cand) > 0:
                    return cand
            # absolute fallback: at least return the nearest
            return np.array([nearest_idx], dtype=int)
        except Exception:
            pass

        # Fallback path (older Shapely): try increasing radii
        for r in (0.25, 1.0, 3.0, 7.0, 15.0, 30.0):
            cand = self._to_positions(self._seg_tree.query(pt.buffer(r)), self._seg_wkb2pos)
            if len(cand) > 0:
                return cand
        # Last resort: no candidates found (shouldn't happen with global data)
        return np.array([], dtype=int)

    def sample_lonlat(self, lon, lat):
        """
        Returns (distance_km (unsigned), country1_id, country2_id)
        """
        pt = Point(lon, lat)

        # 1) country1 (containing)
        c1 = self._country_id_at_lonlat(lon, lat)

        # 2) candidate segments (robust)
        cand_idx = self._candidate_segment_indices(pt)
        if len(cand_idx) == 0:
            # Extremely unlikely: return NaN and keep ids sane for debugging
            return float("nan"), int(c1) if c1 is not None else -1, -1

        # 3) evaluate spherical distance on shortlist
        best_d = 1e18
        best_ids = (None, None)
        ax = self._ax[cand_idx]; ay = self._ay[cand_idx]
        bx = self._bx[cand_idx]; by = self._by[cand_idx]
        ida = self._id_a[cand_idx]; idb = self._id_b[cand_idx]
        for i in range(len(cand_idx)):
            d = greatcircle_point_segment_dist_km(lon, lat, ax[i], ay[i], bx[i], by[i])
            if d < best_d:
                best_d = d
                best_ids = (int(ida[i]), int(idb[i]))

        a, b = best_ids
        # 4) choose country2 as the other side of the nearest segment
        if c1 == a:
            c2 = b
        elif c1 == b:
            c2 = a
        else:
            # If nearest segment doesn't bound c1 (rare tiny-island/topology cases),
            # just return one side; optionally you can refine by testing which side touches c1.
            c2 = b

        return float(best_d), int(c1), int(c2)

    def sample_xyz(self, x, y, z):
        lon, lat = xyz_to_lonlat(x, y, z)
        return self.sample_lonlat(lon, lat)

# ---- core: sample near precomputed border segments ----
def sample_near_border(sampler: BorderSampler, M: int):
    """Return lon, lat, xyz, seg_id, id_a, id_b, r_band for M samples near borders."""
    # segment endpoints in unit vectors
    ax = sampler._ax.astype(np.float64); ay = sampler._ay.astype(np.float64)
    bx = sampler._bx.astype(np.float64); by = sampler._by.astype(np.float64)
    A = lonlat_deg_to_unitvec(ax, ay).astype(np.float64)
    B = lonlat_deg_to_unitvec(bx, by).astype(np.float64)

    # weight segments roughly by angular length (spherical angle)
    dot = np.clip((A * B).sum(axis=1), -1.0, 1.0)
    theta_seg = np.arccos(dot) + 1e-9
    probs = (theta_seg / theta_seg.sum()).astype(np.float64)

    idx = np.random.choice(len(sampler.borders), size=M, p=probs)

    a = A[idx]; b = B[idx]
    # SLERP to random point along segment
    dot_ab = np.clip((a * b).sum(axis=1), -1.0, 1.0)
    theta = np.arccos(dot_ab)
    t = np.random.rand(M)
    sin_th = np.sin(theta)
    coef_a = np.sin((1.0 - t) * theta) / sin_th
    coef_b = np.sin(t * theta) / sin_th
    p = (a * coef_a[:, None] + b * coef_b[:, None])
    p /= np.linalg.norm(p, axis=1, keepdims=True)

    # great-circle tangent at p: tgc = normalize(n × p), with n = normalize(a × b)
    n = np.cross(a, b); n /= np.linalg.norm(n, axis=1, keepdims=True)
    tgc = np.cross(n, p); tgc /= np.linalg.norm(tgc, axis=1, keepdims=True)

    # side vector orthogonal in tangent plane: u = normalize(tgc × p), flip side 50/50
    u = np.cross(tgc, p); u /= np.linalg.norm(u, axis=1, keepdims=True)
    side = (np.random.rand(M) < 0.5).astype(np.float64)
    side = np.where(side > 0, 1.0, -1.0)[:, None]
    u = (u * side)

    # offsets
    d_km_hint, r_band = sample_distance_km(M)
    theta_off = (d_km_hint.astype(np.float64) / R_EARTH_KM)
    p_off = move_along_geodesic(p.astype(np.float32), u.astype(np.float32), theta_off.astype(np.float32))

    lon, lat = unitvec_to_lonlat(p_off)
    seg_id = idx.astype(np.int32)
    id_a = sampler._id_a[idx].astype(np.int32)
    id_b = sampler._id_b[idx].astype(np.int32)
    return lon, lat, p_off, seg_id, id_a, id_b, r_band

# ---- deterministic hashing -> split bucket ----
def _hash_bucket(lon: float, lat: float, split_probs=(0.8, 0.1, 0.1)) -> int:
    """Return 0=train,1=val,2=test using a stable hash of (lon,lat)."""
    p_train, p_val, p_test = split_probs
    # fallback hash if xxhash missing
    s = f"{lon:.6f}|{lat:.6f}"
    if xxhash is None:
        h = hash(s) % 10_000
    else:
        h = xxhash.xxh64(s).intdigest() % 10_000
    th_train = int(p_train * 10_000)
    th_val = int((p_train + p_val) * 10_000)
    return 0 if h < th_train else (1 if h < th_val else 2)

# ---- main entry: build dataset and write Parquet shards ----
def make_dataset(
    sampler: BorderSampler,
    n_total: int,
    out_dir: str = "dataset_dev",
    split_probs=(0.8, 0.1, 0.1),
    mixture=(0.70, 0.30),         # (near_border, uniform_sphere)
    shard_size: int = 500_000,
):
    out_dir = os.path.join(FOLDER_PATH, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    n_border = int(mixture[0] * n_total)
    n_uniform = n_total - n_border

    # 1) draw points
    lon_b, lat_b, xyz_b, seg_id, id_a, id_b, r_band = sample_near_border(sampler, n_border)
    xyz_u = rand_unit_vec(n_uniform)
    lon_u, lat_u = unitvec_to_lonlat(xyz_u)

    lon = np.concatenate([lon_b, lon_u]).astype(np.float32)
    lat = np.concatenate([lat_b, lat_u]).astype(np.float32)
    xyz = np.vstack([xyz_b, xyz_u]).astype(np.float32)
    is_border = np.concatenate([np.ones(n_border, np.uint8), np.zeros(n_uniform, np.uint8)])
    r_band_full = np.concatenate([r_band, np.full(n_uniform, 255, np.uint8)])  # 255 = 'uniform'

    # 2) label with sampler
    N = len(lon)
    dist = np.empty(N, dtype=np.float32)
    c1   = np.empty(N, dtype=np.int32)
    c2   = np.empty(N, dtype=np.int32)

    for i in trange(N, desc="Labeling"):
        d, ci, cj = sampler.sample_lonlat(float(lon[i]), float(lat[i]))
        dist[i] = d; c1[i] = ci; c2[i] = cj

    # 3) deterministic split & write shards
    split = np.fromiter((_hash_bucket(float(lon[i]), float(lat[i]), split_probs) for i in range(N)),
                        dtype=np.uint8, count=N)

    for split_id, name in [(0, "train"), (1, "val"), (2, "test")]:
        sel = (split == split_id)
        if not np.any(sel):
            continue
        df = pd.DataFrame({
            "lon": lon[sel], "lat": lat[sel],
            "x": xyz[sel, 0], "y": xyz[sel, 1], "z": xyz[sel, 2],
            "dist_km": dist[sel].astype(np.float32),
            "c1_id": c1[sel].astype(np.int32),
            "c2_id": c2[sel].astype(np.int32),
            "is_border": is_border[sel],
            "r_band": r_band_full[sel],
        })
        out_split = os.path.join(out_dir, name)
        os.makedirs(out_split, exist_ok=True)
        # shard
        total = len(df); parts = (total + shard_size - 1) // shard_size
        for p in range(parts):
            lo = p * shard_size; hi = min((p + 1) * shard_size, total)
            table = pa.Table.from_pandas(df.iloc[lo:hi], preserve_index=False)
            pq.write_table(table, os.path.join(out_split, f"part-{p:05d}.parquet"), compression="zstd")

    print(f"Wrote Parquet shards to {out_dir}/{{train,val,test}}")

def sample_near_border_for_sampler(sampler, M: int):
    """Use the sampler's preloaded border arrays; returns lon, lat, xyz, is_border_flag, r_band."""
    ax = sampler._ax.astype(np.float64); ay = sampler._ay.astype(np.float64)
    bx = sampler._bx.astype(np.float64); by = sampler._by.astype(np.float64)

    A = lonlat_deg_to_unitvec(ax, ay).astype(np.float64)
    B = lonlat_deg_to_unitvec(bx, by).astype(np.float64)

    dot = np.clip((A * B).sum(axis=1), -1.0, 1.0)
    theta_seg = np.arccos(dot) + 1e-9
    probs = (theta_seg / theta_seg.sum()).astype(np.float64)

    idx = np.random.choice(len(sampler.borders), size=M, p=probs)
    a = A[idx]; b = B[idx]

    # SLERP along segment
    dot_ab = np.clip((a * b).sum(axis=1), -1.0, 1.0)
    theta = np.arccos(dot_ab)
    t = np.random.rand(M)
    sin_th = np.sin(theta)
    coef_a = np.sin((1.0 - t) * theta) / sin_th
    coef_b = np.sin(t * theta) / sin_th
    p = (a * coef_a[:, None] + b * coef_b[:, None])
    p /= np.linalg.norm(p, axis=1, keepdims=True)

    # tangent + side
    n = np.cross(a, b); n /= np.linalg.norm(n, axis=1, keepdims=True)
    tgc = np.cross(n, p); tgc /= np.linalg.norm(tgc, axis=1, keepdims=True)
    u = np.cross(tgc, p); u /= np.linalg.norm(u, axis=1, keepdims=True)
    side = np.where(np.random.rand(M) < 0.5, 1.0, -1.0)[:, None]
    u = (u * side)

    d_km_hint, r_band = sample_distance_km(M)
    theta_off = (d_km_hint.astype(np.float64) / R_EARTH_KM)
    p_off = move_along_geodesic(p.astype(np.float32), u.astype(np.float32), theta_off.astype(np.float32))

    lon, lat = unitvec_to_lonlat(p_off)
    xyz = p_off.astype(np.float32)
    is_border = np.ones(M, np.uint8)
    return lon, lat, xyz, is_border, r_band

# ---------- worker ----------
def _label_and_write_chunk(args):
    """
    Worker entrypoint. Construct a BorderSampler in the child process,
    label the provided lon/lat points, and write a Parquet shard.
    """
    (gpkg_path, layer, id_field, borders_fgb,
     lon_chunk, lat_chunk, xyz_chunk, is_border_chunk, r_band_chunk,
     out_path) = args

    # Rebuild sampler in the child proc
    sampler = BorderSampler(
        gpkg_path=gpkg_path,
        countries_layer=layer,
        id_field=id_field,
        borders_fgb=borders_fgb,
    )

    N = len(lon_chunk)
    dist = np.empty(N, dtype=np.float32)
    c1   = np.empty(N, dtype=np.int32)
    c2   = np.empty(N, dtype=np.int32)

    for i in range(N):
        d, ci, cj = sampler.sample_lonlat(float(lon_chunk[i]), float(lat_chunk[i]))
        dist[i] = d; c1[i] = ci; c2[i] = cj

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
    pq.write_table(table, out_path, compression="zstd")
    return out_path

# ---------- public API ----------
def make_dataset_parallel(
    n_total: int,
    out_dir: str = "dataset_dev_parallel",
    split_probs=(0.8, 0.1, 0.1),
    mixture=(0.70, 0.30),      # (near_border, uniform)
    chunk_size: int = 200_000, # per-process chunk size (tune to RAM/cores)
    max_workers: int | None = None,
    gpkg_path=GPKG_PATH,
    countries_layer=LAYER,
    id_field=ID_FIELD,
    borders_fgb=BORDERS_FGB,
):
    out_dir = os.path.join(FOLDER_PATH, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # We’ll build each split independently to avoid shipping huge arrays between processes
    split_names = ["train", "val", "test"]
    split_counts = [int(split_probs[i] * n_total) for i in range(3)]
    # fix rounding so total matches
    diff = n_total - sum(split_counts)
    for i in range(diff): split_counts[i % 3] += 1

    # A small sampler just for **drawing** border-biased points (no labeling in parent)
    sampler_for_drawing = BorderSampler(
        gpkg_path=gpkg_path, countries_layer=countries_layer,
        id_field=id_field, borders_fgb=borders_fgb
    )

    if max_workers is None:
        try:
            max_workers = max(1, os.cpu_count() - 1)
        except Exception:
            max_workers = 4

    for split_name, N in zip(split_names, split_counts):
        if N == 0:
            continue
        out_split = os.path.join(out_dir, split_name)
        os.makedirs(out_split, exist_ok=True)

        n_border = int(mixture[0] * N)
        n_uniform = N - n_border

        # Draw points (fast, in parent)
        lon_b, lat_b, xyz_b, is_border_b, r_band_b = sample_near_border_for_sampler(sampler_for_drawing, n_border)
        xyz_u = rand_unit_vec(n_uniform)
        lon_u, lat_u = unitvec_to_lonlat(xyz_u)
        is_border_u = np.zeros(n_uniform, np.uint8)
        r_band_u = np.full(n_uniform, 255, np.uint8)

        lon = np.concatenate([lon_b, lon_u]).astype(np.float32)
        lat = np.concatenate([lat_b, lat_u]).astype(np.float32)
        xyz = np.vstack([xyz_b, xyz_u]).astype(np.float32)
        is_border = np.concatenate([is_border_b, is_border_u])
        r_band = np.concatenate([r_band_b, r_band_u])

        # Shuffle within split to avoid any ordering bias
        perm = np.random.permutation(len(lon))
        lon, lat, xyz, is_border, r_band = lon[perm], lat[perm], xyz[perm], is_border[perm], r_band[perm]

        # Create work items (chunks)
        num_chunks = math.ceil(len(lon) / chunk_size)
        futures = []
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            for ci in range(num_chunks):
                lo = ci * chunk_size
                hi = min((ci + 1) * chunk_size, len(lon))
                out_path = os.path.join(out_split, f"part-{ci:05d}.parquet")
                args = (
                    gpkg_path, countries_layer, id_field, borders_fgb,
                    lon[lo:hi], lat[lo:hi], xyz[lo:hi], is_border[lo:hi], r_band[lo:hi],
                    out_path
                )
                futures.append(ex.submit(_label_and_write_chunk, args))

            if tqdm:
                for f in tqdm(as_completed(futures), total=len(futures), desc=f"Writing {split_name}"):
                    _ = f.result()
            else:
                for f in as_completed(futures):
                    _ = f.result()

        print(f"✔ Wrote {num_chunks} shards to {out_split}")

tests = [
    ("Mid-Atlantic ocean (0°N, 0°E)",            0.0,     0.0),
    ("Luxembourg City, LU",                      6.13,   49.61),
    ("Paris, FR (interior; coast is nearest border)", 2.3522, 48.8566),
    ("On US-Mexico border (El Paso/Juárez)",   -106.455, 31.774),
    ("Tokyo, JP (near coast)",                 139.767, 35.681),
    ("Cape Town, ZA (on/near coast)",           18.424, -33.925),
    ("Near the EU triple-point (BE-DE-NL)",      6.0219, 50.7576),
    ("Maseru, Lesotho (enclave inside SA)",     27.486, -29.315),
]

def main():
    start = time.time()
    #sampler = BorderSampler() # -> stable init time of 15~16s on Macbook Pro M4
    # e.g., ~800k total points split into train/val/test
    start = time.time()
    #make_dataset(sampler, n_total=800, out_dir="dataset_dev")
    make_dataset_parallel(
        n_total=800,
        out_dir="dataset_dev_parallel",
        chunk_size=150,     # tune to your RAM; 150k–300k is a good start on M4
        max_workers=None,       # defaults to (CPU cores - 1)
    )
    end = time.time()
    print(end - start, "s for 800")
    # For your main paper results later:
    # make_dataset(sampler, n_total=6_500_000, out_dir="dataset_main")
    #for name, lon, lat in tests:
    #    start = time.time()
    #    d_km, c1, c2 = sampler.sample_lonlat(lon, lat)
    #    end = time.time()
    #    print(end - start)
        #print(f"{name:40s} -> dist≈{d_km:.6f} km, c1={c1}, c2={c2}")
    #end = time.time()
    #print(end - start)
    
main()

'''
Mid-Atlantic ocean (0°N, 0°E)            -> dist≈572.926241 km, c1=289, c2=136
Luxembourg City, LU                      -> dist≈11.549294 km, c1=102, c2=93
Paris, FR                                -> dist≈149.213751 km, c1=93, c2=289
On US-Mexico border (El Paso/Juárez)     -> dist≈1.114023 km, c1=143, c2=240
Tokyo, JP (near coast)                   -> dist≈4.974044 km, c1=2, c2=289
Cape Town, ZA (on/near coast)            -> dist≈1.381645 km, c1=77, c2=289
Near the EU triple-point (BE-DE-NL)      -> dist≈0.128265 km, c1=92, c2=105
Maseru, Lesotho (enclave inside SA)      -> dist≈1.421712 km, c1=65, c2=77
'''