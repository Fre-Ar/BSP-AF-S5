# src/geodata/sampler/sampling.py

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.prepared import prep
from scipy.spatial import KDTree

from utils.utils_geo import (
    read_gdf,
    lonlat_to_unitvec,
    unitvec_to_lonlat,
    normalize_vec,
    arc_segment_attrs,
    _theta_point_to_short_arc,
    R_EARTH_KM,
    GPKG_PATH,
    BORDERS_FGB_PATH,
    COUNTRIES_LAYER,
    ID_FIELD,
)

# --------------------------- sampling logic -------------------------

# TODO: Choose better bands and probabilities
def _sample_distance_km_test(batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    '''
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
    '''
    # Bands low/high (km). Use a small >0 lower bound for r0 to avoid log(0).
    lows  = np.array([0.01,  1.0,   5.0,   10.0,  25.0, 50.0, 150.0], dtype=np.float32)
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
    '''
    Piecewise log-uniform bands favoring near-border points.

    Bands (r = 0..2):
      r0:  [0.01, 10)   km
      r1:  [10,   50)   km
      r2:  [50,  300)   km

    Sampling probabilities (in percentages):
      p ∝ {60, 30, 10}

    Note
    ----
    The r=255 ("uniform globe") bucket is handled elsewhere (purely uniform sampling).

    Returns
    -------
    d : ndarray
        Distances in kilometers (float32) of length `batch_size`.
    bands : ndarray
        Band indices in [0, 1, 2] (uint8).
    '''
    bands = rng.choice([0, 1, 2], size=batch_size, p=[0.6, 0.3, 0.1])
    low  = np.array([0.1, 10.0, 50.0], dtype=np.float32)[bands]
    high = np.array([10.0, 50.0, 300.0], dtype=np.float32)[bands]
    
    # Log-uniform sampling in each band
    u = rng.random(batch_size, dtype=np.float32)
    d = np.exp(np.log(low) + u * (np.log(high) - np.log(low))).astype(np.float32)
    
    return d, bands.astype(np.uint8)



class BorderSampler:
    """
    Spatial labeller for border distance and country IDs.

    Responsibilities
    ----------------
    - Load country polygons (GeoPackage) and prepare them for fast `covers()` queries.
      Used to compute c1_id (containing country).
    - Load pre-segmentized border arcs (FlatGeobuf) and precompute:
        * endpoints on the unit sphere (A3, B3),
        * great-circle normals N3,
        * short-arc angles theta_ab,
        * midpoints M3 on the sphere.
    - Build a KDTree on midpoints M3 for nearest-border shortlist.
    - Provide:
        * `_nearest_segment_vectorized(lon, lat)` -> nearest border distance + (id_a, id_b)
        * `sample_lonlat(lon, lat)` -> dist_km, c1_id, c2_id

    Parameters
    ----------
    gpkg_path : str
        Path to GeoPackage with country polygons.
    countries_layer : str
        Layer name for country polygons inside the GeoPackage.
    id_field : str
        Column name for the country ID.
    borders_fgb : str
        Path to FlatGeobuf with border segments (columns: id_a, id_b, ax, ay, bx, by).
    knn_k : int, optional
        Primary KNN candidate count for nearest-segment search.
    knn_expand : int, optional
        Secondary candidate count for optional expansion (if needed).
    expand_rel : float, optional
        Expansion heuristic: if kth chord <= expand_rel * best chord, re-query with knn_expand.
    """

    def __init__(
        self,
        gpkg_path: str = GPKG_PATH,
        countries_layer: str = COUNTRIES_LAYER,
        id_field: str = ID_FIELD,
        borders_fgb: str = BORDERS_FGB_PATH,
        knn_k: int = 128,              # default candidate budget (tweakable)
        knn_expand: int = 256,         # one-shot expansion if needed
        expand_rel: float = 1.05,      # expand if kth chord <= expand_rel * best chord
    ):

        self.id_field = id_field
        self.knn_k = int(knn_k)
        self.knn_expand = int(knn_expand)
        self.expand_rel = float(expand_rel)

        # Countries (WGS84 lon/lat)
        # -------------------------------------------------
        # 1) Load countries in WGS84 lon/lat (EPSG:4326)
        # -------------------------------------------------
        cdf = read_gdf(
            path=gpkg_path,
            layer=countries_layer,
            id_field=id_field,
            target_crs=4326,
        )
        self.countries = cdf

        # Prepared geometries => fast covers() checks
        self._id2geom_prepared: dict[int, any] = {}
        for _, row in self.countries.iterrows():
            cid = int(row[id_field])
            self._id2geom_prepared[cid] = prep(row.geometry)

        # -------------------------------------------------
        # 2) Load border segments (FlatGeobuf with segmentized shared boundaries)
        # -------------------------------------------------
        bdf = read_gdf(
            path=borders_fgb,
            layer=None,
            id_field=None,
            target_crs=4326,
        )
        self.borders = bdf

        # Fast access arrays for distance loop
        self._ax = self.borders["ax"].to_numpy()
        self._ay = self.borders["ay"].to_numpy()
        self._bx = self.borders["bx"].to_numpy()
        self._by = self.borders["by"].to_numpy()
        self._id_a = self.borders["id_a"].to_numpy(dtype=int)
        self._id_b = self.borders["id_b"].to_numpy(dtype=int)

        # -------------------------------------------------
        # 3) Precompute segment geometry (A3, B3, N3, theta_ab, M3) for fast distances
        # -------------------------------------------------
        A3, B3, N3, theta_ab, M3, _ = arc_segment_attrs(
            self._ax, self._ay, self._bx, self._by
        )

        # KD-tree on midpoints
        self._mid_tree = KDTree(M3, balanced_tree=True)
        self._A3 = A3
        self._B3 = B3
        self._N3 = N3
        self._theta_ab = theta_ab

    # ---- helpers

    def _country_id_at_lonlat(self, lon, lat) -> int | None:
        """
        Find containing country ID at (lon, lat) via prepared geometry coverage.

        Notes
        -----
        This is a simple linear scan over all countries; with a few hundred polygons
        this is usually acceptable and avoids building additional spatial indices.
        """
        pt = Point(lon, lat)
        for cid, pgeom in self._id2geom_prepared.items():
            if pgeom.covers(pt):
                return cid
        return None

    def _knn_candidate_indices(self, p: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        KNN on unit midpoints.

        Parameters
        ----------
        p : ndarray
            Query point on the unit sphere, shape (3,).
        k : int
            Number of nearest neighbors to query (upper-bounded by tree size).

        Returns
        -------
        idx : ndarray
            Indices of candidate segments (int64).
        d : ndarray
            Euclidean chord distances to midpoints (float64).

        Notes
        -----
        For unit vectors, chord distance d and angular distance α relate via:
            d = 2 * sin(α / 2).
        """
        d, idx = self._mid_tree.query(p, k=min(k, self._mid_tree.n), workers=1)
        if np.isscalar(d):  # normalize to 1D arrays
            d = np.array([d], dtype=np.float64)
            idx = np.array([idx], dtype=np.int64)
        return idx.astype(np.int64), d.astype(np.float64)

    def _nearest_segment_vectorized(self, lon: float, lat: float) -> tuple[float, int, int]:
        """
        Nearest short-arc border segment at (lon, lat).

        Uses:
        - KDTree on segment midpoints for a candidate shortlist,
        - vectorized spherical distance kernel `_theta_point_to_short_arc`
          for the candidate set,
        - optional expansion if many segments are similarly close.

        Parameters
        ----------
        lon, lat : float
            Coordinates in degrees (WGS84).

        Returns
        -------
        best_d : float
            Distance in kilometers (unsigned) to the nearest border segment.
        ida : int
            Country ID on one side of the nearest border segment.
        idb : int
            Country ID on the other side of the nearest border segment.
        """
        # Convert query to unit vector on sphere
        p = lonlat_to_unitvec(
            np.array([lon], dtype=np.float64),
            np.array([lat], dtype=np.float64),
        )[0]  # shape (3,)

        idx, chord = self._knn_candidate_indices(p, self.knn_k)
        A = self._A3[idx]
        B = self._B3[idx]
        N = self._N3[idx]
        th = self._theta_ab[idx]

        # Exact spherical distance from p to each short arc
        theta = _theta_point_to_short_arc(p, A, B, N, th)

        best_i = int(np.argmin(theta))
        best_theta = float(theta[best_i])
        best_idx = int(idx[best_i])
        best_d = float(R_EARTH_KM * best_theta)

        # Heuristic expansion:
        # If the furthest chord among the K candidates is not much larger than
        # the best chord, it suggests that there may be multiple segments at
        # similar distance; re-query with a larger candidate set once.
        kth = float(np.max(chord)) 
        if kth <= self.expand_rel * float(chord[best_i]) and self.knn_expand > self.knn_k:
            idx2, _ = self._knn_candidate_indices(p, self.knn_expand)
            uni = np.unique(idx2)
            A = self._A3[uni]
            B = self._B3[uni]
            N = self._N3[uni]
            th = self._theta_ab[uni]

            theta2 = _theta_point_to_short_arc(p, A, B, N, th)
            
            b2 = int(np.argmin(theta2))
            if float(theta2[b2]) < best_theta:
                best_theta = float(theta2[b2])
                best_idx = int(uni[b2])
                best_d = float(R_EARTH_KM * best_theta)

        ida = int(self._id_a[best_idx])
        idb = int(self._id_b[best_idx])
        return best_d, ida, idb

    # ---- public labeling ----

    def sample_lonlat(self, lon: float, lat: float) -> tuple[float, int, int]:
        """
        Labels a single point on the globe.

        Returns
        -------
        best_d : float
            Distance to nearest border segment in kilometers (unsigned).
        c1_id : int
            Containing country ID (or -1 if none found).
        c2_id : int
            Other country ID on the nearest border (never -1).
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
    Draws M points near borders.

    Strategy
    --------
    1) Sample border segments with probability proportional to arc length.
    2) SLERP uniformly along each segment to pick a base point on the border.
    3) Build an orthonormal local frame (tangent along border, normal in tangent plane).
    4) Sample a log-uniform radial offset distance and random side (±1) in the tangent plane.
    5) Move the point along the geodesic in direction of that offset.

    Dumbing it down, 3), 4) and 5) just mean "offset by a small random geodesic distance into a random side".
    
    Returns
    -------
    lon, lat : ndarray
        Longitudes and latitudes (degrees) of samples, shape (M,).
    xyz : ndarray
        Unit vectors on the sphere, shape (M,3).
    is_border : ndarray
        All ones (uint8), marking these as near-border samples.
    r_band : ndarray
        Radial bands (0,1,2) for each point (uint8).
    d_km_hint : ndarray
        Offset distances in km used when constructing the sample (float32).
    id_a, id_b : ndarray
        Country IDs on either side of the segment from which the sample was drawn (int32).
    """
    A = sampler._A3
    B = sampler._B3
    
    # 1) Sample segments proportional to short-arc length
    theta = sampler._theta_ab
    tiny = theta < 1e-9 # mask out tiny/degenerate arcs up front
    probs = np.where(tiny, 0.0, theta) # probability distribution based on arc length
    probs = probs / probs.sum() # make the probability distribution sum to 1
    
    # randomly choose segments based on the probability
    seg_idx = rng.choice(len(theta), size=M, p=probs) 
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
    p = normalize_vec(p)

    # 3) Local frame at p:
    #    - tgc: tangent along border (direction of the short arc),
    #    - u  : outward normal perpendicular in tangent plane (points away from border).
    tgc = np.cross(N, p)                            
    tgc = normalize_vec(tgc)
    
    u = np.cross(tgc, p)                            
    u = normalize_vec(u)
    
    side = rng.integers(0, 2, size=M) * 2 - 1        # ±1
    u = u * side[:, None]
    
    # 4) Sample offsets (km) and convert to angular offsets
    d_km_hint, r_band = _sample_distance_km(M, rng)  
    theta_off = (d_km_hint.astype(np.float64) / R_EARTH_KM) # angle in radians

    # 5) geodesic move: p_off = p*cos(θ) + u*sin(θ)
    c = np.cos(theta_off)[:, None]
    s = np.sin(theta_off)[:, None]
    p_off = p * c + u * s
    p_off = normalize_vec(p_off)

    lon, lat = unitvec_to_lonlat(p_off)
    xyz = p_off.astype(np.float32)
    is_border = np.ones(M, np.uint8)
    id_a = sampler._id_a[seg_idx].astype(np.int32)
    id_b = sampler._id_b[seg_idx].astype(np.int32)
    
    return (
        lon,
        lat,
        xyz,
        is_border,
        r_band.astype(np.uint8),
        d_km_hint.astype(np.float32),
        id_a,
        id_b,
    )