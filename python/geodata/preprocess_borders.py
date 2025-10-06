# preprocess_borders.py
# pip install geopandas pyogrio shapely numpy

from math import ceil
import time
import numpy as np
import geopandas as gpd
from shapely.geometry import LineString, Point
from shapely.errors import GEOSException

# ---- CONFIG ----
GPKG_PATH     = "python/geodata/world_bank_geodata.gpkg"
LAYER         = "countries"
ID_FIELD      = "id"
OUT_FGB       = "python/geodata/borders.fgb"

SEA_ID        = 289                   # your "sea/water" pseudo-country id
R_EARTH_KM    = 6371.0088
MAX_STEP_KM   = 25.0                  # max sub-segment chord length ~<= 25 km
MIN_ARC_DEG   = 1e-6                  # drop ultra-tiny arcs
DEDUP_DECIMALS = 7                    # coordinate rounding for dedup

# ---- spherical helpers ----

def _safe_norm(v, axis=1, keepdims=True, eps=1e-15):
    n = np.linalg.norm(v, axis=axis, keepdims=keepdims)
    return np.where(n < eps, eps, n)

def _safe_div(v, n, eps=1e-15):
    n = np.where(n < eps, 1.0, n)
    return v / n

def _ll_to_unit(lon_deg, lat_deg):
    lon = np.radians(lon_deg); lat = np.radians(lat_deg)
    cl = np.cos(lat)
    v = np.stack([cl*np.cos(lon), cl*np.sin(lon), np.sin(lat)], axis=-1)
    return _safe_div(v, _safe_norm(v)).astype(np.float64)

def _unit_to_ll(v):
    x, y, z = v[...,0], v[...,1], v[...,2]
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
    return float(lon), float(lat)

def _slerp(a_u, b_u, t):
    """Spherical linear interpolation between two unit vectors a_u, b_u."""
    dot = np.clip(np.dot(a_u, b_u), -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-15:
        return a_u.copy()
    s = np.sin(theta)
    return (np.sin((1.0 - t) * theta) / s) * a_u + (np.sin(t * theta) / s) * b_u

def _num_steps_for_pair(a_ll, b_ll, max_step_km=MAX_STEP_KM):
    """How many spherical steps are needed so that each sub-arc <= max_step_km."""
    a_u = _ll_to_unit(a_ll[0], a_ll[1])[0]
    b_u = _ll_to_unit(b_ll[0], b_ll[1])[0]
    theta = np.arccos(np.clip(np.dot(a_u, b_u), -1.0, 1.0))
    arc_km = R_EARTH_KM * theta
    steps = max(1, ceil(arc_km / max_step_km))
    return steps, a_u, b_u, theta

def _densify_pair_gc(a_ll, b_ll, max_step_km=MAX_STEP_KM):
    """
    Great-circle densification between two lon/lat points (a_ll -> b_ll).
    Returns a list of lon/lat coordinates starting from a_ll and including b_ll.
    """
    steps, a_u, b_u, theta = _num_steps_for_pair(a_ll, b_ll, max_step_km)
    if steps == 1:
        return [a_ll, b_ll]
    out = []
    for t in range(steps):
        tt = t / steps
        u = _slerp(a_u, b_u, tt)
        out.append(_unit_to_ll(_safe_div(u, _safe_norm(u))))
    out.append(b_ll)
    return out

def _segmentize_linestring_gc(ls: LineString, max_step_km=MAX_STEP_KM):
    """Yield consecutive spherical (a,b) lon/lat pairs with sub-arc <= max_step_km."""
    coords = list(ls.coords)
    if len(coords) < 2:
        return
    prev = coords[0]
    for k in range(1, len(coords)):
        chunk = _densify_pair_gc(prev, coords[k], max_step_km)
        for i in range(len(chunk) - 1):
            a = chunk[i]; b = chunk[i+1]
            if a != b:
                yield a, b
        prev = coords[k]

def _pair_features(gdf):
    """Return candidate neighboring pairs by bbox sjoin, reduced to i<j."""
    short = gpd.sjoin(
        gdf[["geometry"]],
        gdf[["geometry"]],
        how="inner",
        predicate="intersects"
    )
    short = short[short.index < short.index_right]
    return list(zip(short.index.to_list(), short.index_right.to_list()))

def _arc_geometry_and_attrs(a_ll, b_ll):
    """Compute spherical attributes for a sub-segment."""
    A3 = _ll_to_unit(a_ll[0], a_ll[1])[0]
    B3 = _ll_to_unit(b_ll[0], b_ll[1])[0]
    # normal and angle
    N3 = np.cross(A3, B3); N3 = _safe_div(N3, _safe_norm(N3))
    cos_ab = float(np.clip(np.dot(A3, B3), -1.0, 1.0))
    theta_ab = float(np.arccos(cos_ab))
    # discard near-zero arcs
    if np.degrees(theta_ab) < MIN_ARC_DEG:
        return None
    # midpoint on great circle (normalized A+B; fallback to A if near-antipodal)
    M = A3 + B3
    nM = np.linalg.norm(M)
    if nM < 1e-12:
        M = A3.copy()
    else:
        M = M / nM
    geom = LineString([a_ll, b_ll])
    return {
        "ax": float(a_ll[0]), "ay": float(a_ll[1]),
        "bx": float(b_ll[0]), "by": float(b_ll[1]),
        "mx": float(M[0]),   "my": float(M[1]),   "mz": float(M[2]),
        "theta_ab": theta_ab,
        "geometry": geom,
    }

def _dedup_df(df):
    # round coordinates to avoid trivial duplicates
    for c in ("ax","ay","bx","by"):
        df[c] = df[c].round(DEDUP_DECIMALS)
    df = df.drop_duplicates(subset=["id_a","id_b","ax","ay","bx","by"]).reset_index(drop=True)
    return df

# ---- MAIN PIPELINE ----

def create_borders():
    # 1) Load countries (EPSG:4326)
    gdf = gpd.read_file(GPKG_PATH, layer=LAYER)
    if gdf.crs is None or (gdf.crs.to_epsg() or 4326) != 4326:
        gdf = gdf.to_crs(4326)
    gdf = gdf[[ID_FIELD, "geometry"]].reset_index(drop=True)

    # 2) Neighbor pairs by bbox; confirm touches; extract shared boundary
    pairs = _pair_features(gdf)
    records = []

    for i, j in pairs:
        geom_i = gdf.geometry.iat[i]
        geom_j = gdf.geometry.iat[j]
        try:
            if not geom_i.touches(geom_j):
                continue
        except GEOSException:
            continue

        inter = geom_i.intersection(geom_j)
        if inter.is_empty:
            continue

        id_i = int(gdf.iat[i, gdf.columns.get_loc(ID_FIELD)])
        id_j = int(gdf.iat[j, gdf.columns.get_loc(ID_FIELD)])
        if id_i == id_j:
            continue  # same country (multiparts), skip

        id_a, id_b = (id_i, id_j) if id_i < id_j else (id_j, id_i)

        # Segmentize each shared line **on the sphere**
        def _iter_lines(g):
            if g.is_empty:
                return
            gt = g.geom_type
            if gt == "LineString":
                yield g
            elif gt == "MultiLineString":
                for sub in g.geoms:
                    if not sub.is_empty:
                        yield sub
            elif hasattr(g, "geoms"):  # geometry collection
                for sub in g.geoms:
                    yield from _iter_lines(sub)

        for line in _iter_lines(inter):
            if line.is_empty or line.length == 0:
                continue
            for a_ll, b_ll in _segmentize_linestring_gc(line, MAX_STEP_KM):
                attrs = _arc_geometry_and_attrs(a_ll, b_ll)
                if attrs is None:
                    continue
                attrs.update({"id_a": id_a, "id_b": id_b})
                records.append(attrs)

    # 3) Coastlines: dissolve countries -> boundary lines -> assign to touching country, pair with SEA_ID
    if not records:
        # Keep going; maybe coastlines will populate
        pass

    dissolved = gdf.dissolve(by=None)  # single row, unary_union geometry
    world_outline = dissolved.geometry.boundary  # LineString/MultiLineString
    if not world_outline.is_empty:
        # spatial index for country lookup by midpoint
        countries_idx = gdf.sindex

        def _country_at_point(pt: Point) -> int | None:
            # fast bbox hit, then precise contains/covers
            try:
                cand = list(countries_idx.query(pt))
            except Exception:
                cand = []
            for idx in cand:
                geom = gdf.geometry.iat[idx]
                if geom.covers(pt):
                    return int(gdf.iat[idx, gdf.columns.get_loc(ID_FIELD)])
            return None

        def _iter_lines_outline(g):
            if g.is_empty:
                return
            if g.geom_type == "LineString":
                yield g
            elif g.geom_type == "MultiLineString":
                for sub in g.geoms:
                    if not sub.is_empty:
                        yield sub
            elif hasattr(g, "geoms"):
                for sub in g.geoms:
                    yield from _iter_lines_outline(sub)

        for line in _iter_lines_outline(world_outline):
            for a_ll, b_ll in _segmentize_linestring_gc(line, MAX_STEP_KM):
                attrs = _arc_geometry_and_attrs(a_ll, b_ll)
                if attrs is None:
                    continue
                # Decide which country owns this coastline segment via midpoint
                mid_lon = 0.5*(attrs["ax"] + attrs["bx"])
                mid_lat = 0.5*(attrs["ay"] + attrs["by"])
                cid = _country_at_point(Point(mid_lon, mid_lat))
                if cid is None:
                    continue  # could be tiny slivers; skip
                id_a, id_b = (cid, SEA_ID) if cid < SEA_ID else (SEA_ID, cid)
                attrs.update({"id_a": id_a, "id_b": id_b})
                records.append(attrs)

    if not records:
        raise RuntimeError("No border or coastline segments were created. Check input geometries/CRS.")

    borders = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)

    # 4) Dedup and cleanup
    borders = _dedup_df(borders)
    # drop any arcs that are still near-zero by angle
    borders = borders[borders["theta_ab"] > np.radians(MIN_ARC_DEG)].reset_index(drop=True)

    # 5) Save as FlatGeobuf
    # Columns expected by the runtime: id_a, id_b, ax, ay, bx, by
    # Extras for KNN speed: mx,my,mz (midpoint unit vector), theta_ab (short-arc angle)
    borders.to_file(OUT_FGB, driver="FlatGeobuf")
    print(f"Saved {len(borders):,} segments to {OUT_FGB}")


if __name__ == "__main__":
    t0 = time.time()
    create_borders()
    print(f"Preprocessing took {time.time() - t0:.1f}s")
