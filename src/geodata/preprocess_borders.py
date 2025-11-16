# src/geodata/preprocess_borders.py

from math import ceil
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.errors import GEOSException

from utils.utils import _slerp
from utils.utils_geo import (
    R_EARTH_KM,
    lonlat_to_unitvec,
    unitvec_to_lonlat,
    normalize_vec,
    read_gdf,
    arc_segment_attrs, 
)

# ---- CONFIG ----
MAX_STEP_KM   = 25.0                  # max sub-segment chord length ~<= 25 km
MIN_ARC_DEG   = 1e-6                  # drop ultra-tiny arcs
DEDUP_DECIMALS = 7                    # coordinate rounding for dedup

# ---- spherical helpers ----

def _num_steps_for_pair(a_ll, b_ll, max_step_km=MAX_STEP_KM):
    """
    Computes how many great-circle steps are needed so that each sub-arc <= max_step_km.
    
    Parameters
    ----------
    a_ll : tuple
        Start point (lon, lat) in degrees.
    b_ll : tuple
        End point (lon, lat) in degrees.
    max_step_km : float, optional
        Maximum allowed length of each sub-segment along the great-circle arc.

    Returns
    -------
    steps : int
        Number of segments to split the arc into (at least 1).
    a_u : ndarray
        Unit vector for `a_ll` (shape (3,)).
    b_u : ndarray
        Unit vector for `b_ll` (shape (3,)).
    theta : float
        Great-circle angle between `a_ll` and `b_ll` in radians.
    """
    # convert points to unit vectors
    a_u = lonlat_to_unitvec(a_ll[0], a_ll[1])[0]
    b_u = lonlat_to_unitvec(b_ll[0], b_ll[1])[0]
    # compute angle between them
    theta = np.arccos(np.clip(np.dot(a_u, b_u), -1.0, 1.0))
    # compute arc length between them
    arc_km = R_EARTH_KM * theta
    # divide the arc length into as few sections as possible.
    steps = max(1, ceil(arc_km / max_step_km))
    return steps, a_u, b_u, theta

def _densify_pair_gc(a_ll, b_ll, max_step_km=MAX_STEP_KM):
    """
    Densifies a great-circle arc between two lon/lat points.

    Parameters
    ----------
    a_ll : tuple
        Start point (lon, lat) in degrees.
    b_ll : tuple
        End point (lon, lat) in degrees.
    max_step_km : float, optional
        Maximum allowed length of each sub-segment in km.

    Returns
    -------
    coords : list[tuple]
        List of (lon, lat) points along the great-circle, starting at `a_ll`
        and including `b_ll`. Consecutive pairs satisfy the step-length bound.

    Notes
    -----
    HOW:
    - Compute how many steps are needed with `_num_steps_for_pair`.
    - Use `_slerp` (spherical linear interpolation) in 3D between `a_u` and `b_u`
      at fractions t = 0/steps, ..., (steps-1)/steps.
    - Convert each interpolated unit vector back to lon/lat.
    - Append the exact original endpoint `b_ll`.
    """
    # Compute how many steps needed
    steps, a_u, b_u, theta = _num_steps_for_pair(a_ll, b_ll, max_step_km)
    # if one, the arc is already small enough.
    if steps == 1:
        return [a_ll, b_ll]
    
    # create the points in between
    out = []
    for t in range(steps):
        # Interpolate on the unit sphere
        tt = t / steps
        u = _slerp(a_u, b_u, tt)
        # normalize for safety
        out.append(unitvec_to_lonlat(normalize_vec(u)))
    out.append(b_ll)
    return out

def _segmentize_linestring_gc(ls: LineString, max_step_km=MAX_STEP_KM):
    """
    Yields consecutive great-circle (a, b) lon/lat pairs with bounded sub-arc length.

    Parameters
    ----------
    ls : shapely.geometry.LineString
        Input line in lon/lat space (EPSG:4326) describing a border or coastline.
    max_step_km : float, optional
        Maximum allowed length of each sub-segment in km.

    Yields
    ------
    a_ll, b_ll : tuple, tuple
        Consecutive lon/lat endpoints of each spherical sub-segment, with
        great-circle length <= max_step_km.
    """
    coords = list(ls.coords)
    if len(coords) < 2:
        return
    # Iterate through consecutive coordinates of the linestring.
    prev = coords[0]
    for k in range(1, len(coords)):
        # For each original segment (prev -> coords[k]), densify the pair.
        chunk = _densify_pair_gc(prev, coords[k], max_step_km)
        # emit all consecutive pairs as output segments.
        for i in range(len(chunk) - 1):
            a = chunk[i]; b = chunk[i+1]
            if a != b:
                yield a, b
        prev = coords[k]

def _pair_features(gdf):
    """
    Returns candidate neighboring pairs of polygons using a bbox spatial join.

    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with a 'geometry' column in lon/lat (EPSG:4326).

    Returns
    -------
    pairs : list[tuple]
        List of index pairs (i, j) with i < j, representing polygons whose
        bounding boxes intersect.
    """
    short = gpd.sjoin(
        gdf[["geometry"]],
        gdf[["geometry"]],
        how="inner",
        predicate="intersects"
    )
    # Discard self-pairs and enforce i < j to avoid duplicates.
    short = short[short.index < short.index_right]
    return list(zip(short.index.to_list(), short.index_right.to_list()))


def _dedup_df(df):
    """
    Deduplicate segments by id pair and endpoints, after rounding coordinates.

    Parameters
    ----------
    df : GeoDataFrame
        DataFrame with at least columns:
        ("id_a", "id_b", "ax", "ay", "bx", "by").

    Returns
    -------
    df_out : GeoDataFrame
        Deduplicated DataFrame with coordinates rounded to `DEDUP_DECIMALS`.
    """
    # round coordinates to avoid trivial duplicates
    for c in ("ax","ay","bx","by"):
        df[c] = df[c].round(DEDUP_DECIMALS)
    df = df.drop_duplicates(subset=["id_a","id_b","ax","ay","bx","by"]).reset_index(drop=True)
    return df

# --------------------------- public API -----------------------------

def create_borders(
    path: str,
    out_path: str,
    layer: str,
    id_field: str,
    max_step_km = MAX_STEP_KM,
    min_arc_deg = MIN_ARC_DEG):
    """
    Extracts short spherical border segments (polygon-polygon adjacencies) from a layer 
    and writes them into a FlatGeobuf file at `out_path`.

    Parameters
    ----------
    path : str
        Path to a vector file (e.g. GPKG) containing polygons with id field.
    out_path : str
        Path to output FlatGeobuf file for the border segments.
    layer : str
        Layer name inside the source dataset to read (e.g. 'countries').
    id_field : str
        Name of the attribute column containing unique integer ids per polygon.
    max_step_km : float, optional
        Maximum target length for each sub-segment (in km along the sphere).
        Controls how finely borders are split.
    min_arc_deg : float, optional
        Minimum angular span (in degrees) for a segment to be kept. Shorter arcs
        are dropped as numerical noise.

    Notes
    -----
    HOW:
    1) Load the country layer in EPSG:4326 and keep only (id_field, geometry).
    2) Use `_pair_features` to find bbox-intersecting pairs, then filter to
       polygons that truly `touch()`, and intersect to get shared boundaries.
    3) For each shared boundary, densify on the sphere via `_segmentize_linestring_gc`,
       compute per-segment attributes with `_arc_geometry_and_attrs`, and store
       (id_a, id_b) with id_a < id_b.
    4) For coastlines, dissolve all polygons, take the world boundary, and:
         - densify each coastline segment in spherical space,
         - for each sub-segment, choose the owning country by midpoint,
         - pair that id with `sea_id`.
    5) Deduplicate segments (rounding coordinates) and drop segments below
       `min_arc_deg`.
    6) Save everything to FlatGeobuf. The resulting file is ready for fast
       KD-tree based nearest-border queries in the sampler.
    """
    
    # 1) Load countries (EPSG:4326)
    gdf = read_gdf(path, layer, id_field)

    # 2) Neighbor pairs by bbox; confirm touches; extract shared boundary
    pairs = _pair_features(gdf)

    # accumulate raw segment endpoints + ids
    ax_list, ay_list = [], []
    bx_list, by_list = [], []
    id_a_list, id_b_list = [], []
    
    for i, j in pairs:
        # get candidate geometry
        geom_i = gdf.geometry.iat[i]
        geom_j = gdf.geometry.iat[j]
        
        # Narrow-phase filter: only keep polygons whose boundaries *actually* touch.
        # This discards pairs that merely have intersecting bounding boxes.
        try:
            if not geom_i.touches(geom_j):
                continue
        except GEOSException:
            # Some pathological geometries can throw; we treat them as "not touching"
            continue

        # Compute the shared boundary line(s) between the two polygons.
        inter = geom_i.intersection(geom_j)
        if inter.is_empty:
            # Bboxes intersect and touches() said yes, but intersection is empty:
            # we treat as no usable border (robustness against dirty data).
            continue

        # Extract canonical country ids for both sides of this shared boundary
        id_i = int(gdf.iat[i, gdf.columns.get_loc(id_field)])
        id_j = int(gdf.iat[j, gdf.columns.get_loc(id_field)])
        
        # If both polygons have the same id, it's just different parts of the same country
        if id_i == id_j:
            continue  # we do not want internal borders here.

        # Store (min_id, max_id) to prevent duplication.
        id_a, id_b = (id_i, id_j) if id_i < id_j else (id_j, id_i)

        # Segmentize each shared line **on the sphere**
        # by normalizing the intersection geometry to an iterator over LineStrings.
        def _iter_lines(g):
             
            if g.is_empty:
                return
            gt = g.geom_type
            if gt == "LineString":
                # Single contiguous border line
                yield g
            elif gt == "MultiLineString":
                # Several disjoint border pieces between the same pair of polygons
                for sub in g.geoms:
                    if not sub.is_empty:
                        yield sub
            elif hasattr(g, "geoms"):  # geometry collection
                # Recursively extract any LineStrings contained inside
                for sub in g.geoms:
                    yield from _iter_lines(sub)

        # for each segment
        for line in _iter_lines(inter):
            if line.is_empty or line.length == 0:
                # Degenerate border piece; nothing to segmentize
                continue
            
            # Split this border line into many short great-circle segments whose
            # spherical chord length is <= max_step_km.
            for a_ll, b_ll in _segmentize_linestring_gc(line, max_step_km):
                ax, ay = float(a_ll[0]), float(a_ll[1])
                bx, by = float(b_ll[0]), float(b_ll[1])

                ax_list.append(ax)
                ay_list.append(ay)
                bx_list.append(bx)
                by_list.append(by)
                id_a_list.append(id_a)
                id_b_list.append(id_b)

    if not ax_list:
        raise RuntimeError("No border segments were created. Check input geometries/CRS.")

    # compute arc segement attributes
    ax = np.array(ax_list, dtype=np.float64)
    ay = np.array(ay_list, dtype=np.float64)
    bx = np.array(bx_list, dtype=np.float64)
    by = np.array(by_list, dtype=np.float64)
    id_a_arr = np.array(id_a_list, dtype=np.int32)
    id_b_arr = np.array(id_b_list, dtype=np.int32)

    _A3, _B3, _N3, theta_ab, M3, mask_valid = arc_segment_attrs(
        ax, ay, bx, by, min_arc_deg=min_arc_deg
    )

    # Apply min-arc filter to everything
    if not np.any(mask_valid):
        raise RuntimeError("All segments were below min_arc_deg; nothing to save.")

    # filter segements that are either too small or numerically unstable
    ax = ax[mask_valid]
    ay = ay[mask_valid]
    bx = bx[mask_valid]
    by = by[mask_valid]
    id_a_arr = id_a_arr[mask_valid]
    id_b_arr = id_b_arr[mask_valid]
    theta_ab = theta_ab[mask_valid]
    M3 = M3[mask_valid]

    # 3) Build GeoDataFrame from filtered arrays
    data = {
        "id_a": id_a_arr,
        "id_b": id_b_arr,
        "ax": ax.astype(np.float32),
        "ay": ay.astype(np.float32),
        "bx": bx.astype(np.float32),
        "by": by.astype(np.float32),
        "mx": M3[:, 0].astype(np.float32),
        "my": M3[:, 1].astype(np.float32),
        "mz": M3[:, 2].astype(np.float32),
        "theta_ab": theta_ab.astype(np.float32),
    }

    # build LineString geometry for each segment
    geometries = [LineString([(ax[i], ay[i]), (bx[i], by[i])]) for i in range(len(ax))]
    # Assemble all tiny border segments into a GeoDataFrame in lon/lat (EPSG:4326)
    borders = gpd.GeoDataFrame(data, geometry=geometries, crs=4326)
    
    # 4) Dedup and cleanup
    borders = _dedup_df(borders)

    # 5) Save as FlatGeobuf
    
    # Columns expected by the runtime: id_a, id_b, ax, ay, bx, by
    # Extras for KNN speed: mx,my,mz (midpoint unit vector), theta_ab (short-arc angle)
    borders.to_file(out_path, driver="FlatGeobuf")
    print(f"Saved {len(borders):,} segments to {out_path}")

