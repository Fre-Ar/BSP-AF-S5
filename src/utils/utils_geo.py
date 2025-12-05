# src/utils/utils_geo.py
import geopandas as gpd
import numpy as np
import os

# Optional Numba acceleration
try:
    from numba import njit
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False
    
from .utils import _safe_norm, _safe_div

# ------------------------- CONSTANTS --------------------------

R_EARTH_KM = 6371.0088 # mean Earth radius
MERCATOR_LIMIT = 85.05112878  # Web Mercator latitude cap, degrees

# --------------------------- CONFIG ---------------------------

# Base folder for static geodata
FOLDER_PATH = "src/geodata/data"        

# Input geodata
GPKG_PATH           = os.path.join(FOLDER_PATH, "world_bank_geodata.gpkg")
BORDERS_FGB_PATH    = os.path.join(FOLDER_PATH, "borders.fgb")
ADJACENCY_JSON_PATH = os.path.join(FOLDER_PATH, "geodata_adjacency.json")
COUNTRIES_ECOC_PATH = os.path.join(FOLDER_PATH, "countries.ecoc.json")

# Training data + checkpoints
TRAINING_DATA_PATH = "src/geodata/parquet"   
CHECKPOINT_PATH = "src/checkpoints"     
DATA_ANALYSIS_PATH = "src/analysis"   

# Dataset meta
ECOC_BITS = 32
NUM_COUNTRIES = 298
COUNTRIES_LAYER = "countries"
ID_FIELD    = "id"  

SEED = 42 # global default seed

# -----------------------------
# Math helpers
# -----------------------------
def normalize_vec(v: np.ndarray, axis=1):
    """
    Normalizes vectors to unit length, robustly.

    Parameters
    ----------
    v : ndarray
        Input array of shape (..., 3).

    Returns
    -------
    v_u : ndarray
        Unit vectors of the same shape, with safe handling of very small norms.
    """
    return _safe_div(v, _safe_norm(v, axis=axis))

def lonlat_to_unitvec(lon_deg, lat_deg, dtype=np.float32) -> np.ndarray:
    """
    Converts lon/lat in degrees to unit vectors on the sphere.

    Accepts scalars or arrays; returns shape (..., 3).

    Parameters
    ----------
    lon_deg, lat_deg : array-like or scalar
        Longitudes and latitudes in degrees.
    dtype : data-type, optional
        Output dtype for the vectors (default float32).

    Returns
    -------
    v : ndarray
        Array of unit vectors in R^3, shape (..., 3).
    """
    lon = np.radians(lon_deg)
    lat = np.radians(lat_deg)
    cl  = np.cos(lat)

    v = np.stack([cl * np.cos(lon),
                  cl * np.sin(lon),
                  np.sin(lat)], axis=-1)
    v = normalize_vec(v, axis=-1)

    return v.astype(dtype)

def unitvec_to_lonlat(v: np.ndarray, dtype=np.float32):
    """
    Converts unit vectors on the sphere to lon/lat in degrees.

    Parameters
    ----------
    v : ndarray
        Unit vectors of shape (..., 3).
    dtype : data-type, optional
        Output dtype for lon/lat (default float32).

    Returns
    -------
    lon, lat : ndarray
        Longitudes and latitudes in degrees, shape (...,).
    """
    v = np.asarray(v)
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return lon.astype(dtype), lat.astype(dtype)

def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """
    Returns the angle (radians) between two unit vectors u and v.
    """
    return np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))

def move_along_geodesic(p: np.ndarray, t: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """
    Rotates unit vectors `p` along tangent directions `t` by angles `theta` (radians).

    Geometrically, this moves points along great circles on the sphere.

    Parameters
    ----------
    p : ndarray
        Base points on the sphere, shape (N, 3), unit vectors.
    t : ndarray
        Tangent directions at each point, shape (N, 3), unit vectors orthogonal to `p`.
    theta : ndarray
        Rotation angles in radians, shape (N,).

    Returns
    -------
    out : ndarray
        Rotated unit vectors, shape (N, 3).
    """
    ct = np.cos(theta)[:, None]; st = np.sin(theta)[:, None]
    out = p * ct + t * st
    out = normalize_vec(out)
    return out.astype(np.float32)

# ---------------------------------------------------------------------
# Scalar reference distance: great-circle point-to-segment
# ---------------------------------------------------------------------
def greatcircle_point_segment_dist_km(p_lon, p_lat, a_lon, a_lat, b_lon, b_lat) -> float:
    """
    Exact spherical distance from a point to a short great-circle segment.

    The computation is done on the unit sphere and then scaled by R_EARTH_KM.

    Parameters
    ----------
    p_lon, p_lat : float
        Point coordinates in degrees.
    a_lon, a_lat : float
        First endpoint of the segment in degrees.
    b_lon, b_lat : float
        Second endpoint of the segment in degrees.

    Returns
    -------
    float
        Shortest distance from the point to the segment in kilometers.
    """
    p = lonlat_to_unitvec(p_lon, p_lat, dtype=np.float64)[0]
    a = lonlat_to_unitvec(a_lon, a_lat, dtype=np.float64)[0]
    b = lonlat_to_unitvec(b_lon, b_lat, dtype=np.float64)[0]
    
    n = np.cross(a, b)
    nn = np.linalg.norm(n)
    if nn == 0.0:
        return R_EARTH_KM * min(
            _angle_between(p, a),
            _angle_between(p, b)
        )
    n /= nn

    c = np.cross(n, p)
    c = np.cross(c, n)
    c /= np.linalg.norm(c)

    ab = _angle_between(a, b)
    ac = _angle_between(a, c)
    cb = _angle_between(c, b)

    if abs((ac + cb) - ab) < 1e-10:
        theta = _angle_between(p, c)
    else:
        theta = min(
            _angle_between(p, a),
            _angle_between(p, b)
        )
    return float(R_EARTH_KM * theta)


# ---------------------------------------------------------------------
# Vectorized segment attributes
# ---------------------------------------------------------------------
def arc_segment_attrs(ax, ay, bx, by, min_arc_deg: float | None = None):
    """
    Computes spherical attributes for border segments defined by lon/lat endpoints.

    Parameters
    ----------
    ax, ay : array-like
        Longitudes and latitudes of segment start points (degrees).
    bx, by : array-like
        Longitudes and latitudes of segment end points (degrees).
    min_arc_deg : float, optional
        If provided, segments with angular length below this threshold (in degrees)
        are marked as invalid in the returned mask_valid.

    Returns
    -------
    A3, B3 : ndarray
        Unit direction vectors of endpoints in R^3, shape (S, 3).
    N3 : ndarray
        Unit great-circle normals for each segment, shape (S, 3).
    theta_ab : ndarray
        Short-arc angles between endpoints (radians), shape (S,).
    M3 : ndarray
        Unit midpoint direction vectors on the great-circle (normalized A+B
        with antipodal fallback), shape (S, 3).
    mask_valid : ndarray
        Boolean mask of shape (S,) indicating which segments pass the min_arc_deg
        threshold. If min_arc_deg is None, all True.
    """
    ax = np.asarray(ax, dtype=np.float64)
    ay = np.asarray(ay, dtype=np.float64)
    bx = np.asarray(bx, dtype=np.float64)
    by = np.asarray(by, dtype=np.float64)

    # Endpoints as unit vectors
    A3 = lonlat_to_unitvec(ax, ay).astype(np.float64)  # (S,3)
    B3 = lonlat_to_unitvec(bx, by).astype(np.float64)  # (S,3)

    # Great-circle normals (unit)
    N3 = np.cross(A3, B3)
    N3 = normalize_vec(N3)

    # Short-arc angle
    cos_ab = np.clip((A3 * B3).sum(axis=1), -1.0, 1.0)
    theta_ab = np.arccos(cos_ab)  # (S,)

    # Optional min-arc filter
    if min_arc_deg is not None:
        mask_valid = np.degrees(theta_ab) >= float(min_arc_deg)
    else:
        mask_valid = np.ones_like(theta_ab, dtype=bool)

    # Midpoints for KNN (normalize A+B; fallback to A if near-antipodal)
    M3 = A3 + B3
    nn = _safe_norm(M3)
    M3 = _safe_div(M3, nn)
    near_zero = (nn[:, 0] < 1e-12)
    if np.any(near_zero):
        M3[near_zero] = A3[near_zero]

    return A3, B3, N3, theta_ab, M3, mask_valid

# ---------------------------------------------------------------------
# Vector distance kernel: θ(point, short arc) (NumPy + Numba)
# ---------------------------------------------------------------------
def _theta_point_to_short_arc_numpy(
    p: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    N: np.ndarray,
    theta_ab: np.ndarray) -> np.ndarray:
    """
    NumPy vectorized distance: angle from point p to short arcs A-B.

    Parameters
    ----------
    p : ndarray
        Query point on the unit sphere, shape (3,).
    A, B : ndarray
        Unit endpoints for K candidate segments, shape (K, 3).
    N : ndarray
        Unit great-circle normals for each segment, shape (K, 3).
    theta_ab : ndarray
        Short-arc angles between A and B for each segment, shape (K,).

    Returns
    -------
    theta : ndarray
        Angles (radians) from p to each short arc A-B for K candidates, shape (K,).
    """
    C = np.cross(N, p)
    C = np.cross(C, N)
    C = normalize_vec(C)

    def aco(x): return np.arccos(np.clip(x, -1.0, 1.0))

    ac = aco(np.sum(A * C, axis=1))
    cb = aco(np.sum(C * B, axis=1))
    on_short = np.abs((ac + cb) - theta_ab) < 1e-10

    theta_proj = _angle_between(C, p)       # (K,)
    theta_pa   = _angle_between(A, p)
    theta_pb   = _angle_between(B, p)
    theta_end  = np.minimum(theta_pa, theta_pb)

    return np.where(on_short, theta_proj, theta_end)

if _HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _theta_point_to_short_arc_numba(p, A, B, N, theta_ab):
        """
        Numba-compiled version of `_theta_point_to_short_arc_numpy`.

        All arrays are float64, shapes:
          - p: (3,)
          - A,B,N: (K,3)
          - theta_ab: (K,)
        Returns an array of shape (K,) with angles in radians.
        
        Honestly, most of what's going on here is black magic but, hey, it runs faster.
        """
        K = A.shape[0]
        out = np.empty(K, dtype=np.float64)

        for i in range(K):
            # Project p onto great-circle plane defined by N[i]
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
    
    
if _HAS_NUMBA:
    _theta_point_to_short_arc = _theta_point_to_short_arc_numba
else:
    _theta_point_to_short_arc = _theta_point_to_short_arc_numpy
    
# -----------------------------
# I/O
# -----------------------------

def read_gdf(
    path: str,
    layer: str | None,
    id_field: str | None,
    target_crs: int | str = 4326,
) -> gpd.GeoDataFrame:
    """
    Reads a GeoDataFrame from a file+layer, normalize CRS, and keep only (id_field, geometry).

    Parameters
    ----------
    path : str
        Path to the file (e.g., GeoPackage, FlatGeobuf, Shapefile).
    layer : str or None
        Layer name inside the file. If None, let GeoPandas pick the default.
    id_field : str, optional
        Column name containing the feature ID.
    target_crs : int or str, optional
        Target CRS for the output. If int, treated as EPSG code (e.g. 4326).
        If str, passed directly to `to_crs`.

    Returns
    -------
    gdf : GeoDataFrame
        GeoDataFrame with standardized CRS and only (id_field, geometry) columns.
    """
    gdf = gpd.read_file(path, layer=layer)

    # --- normalize CRS ---
    if gdf.crs is None:
        # No CRS attached: assume target_crs
        gdf = gdf.set_crs(target_crs, allow_override=True)
    else:
        if isinstance(target_crs, int):
            # Compare by EPSG if possible
            epsg = gdf.crs.to_epsg()
            if epsg != target_crs:
                gdf = gdf.to_crs(target_crs)
        else:
            # target_crs is something like "EPSG:4326" or a proj string
            if str(gdf.crs) != str(target_crs):
                gdf = gdf.to_crs(target_crs)

    # Keep only id and geometry
    if id_field is None:
        gdf = gdf.reset_index(drop=True)
    else:
        gdf = gdf[[id_field, "geometry"]].reset_index(drop=True)
    return gdf
