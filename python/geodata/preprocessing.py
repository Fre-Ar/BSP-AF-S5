# preprocess_borders.py
# pip install geopandas pyogrio shapely

from math import ceil
import geopandas as gpd
from shapely.geometry import LineString
from shapely.errors import GEOSException
import time

# ---- CONFIG ----
GPKG_PATH   = "python/geodata/world_bank_geodata.gpkg"
LAYER       = "countries"
ID_FIELD    = "id"                    
OUT_FGB     = "python/geodata/borders.fgb"
MAX_STEP_DEG = 0.25                   # densify so no segment step exceeds this (in degrees)

# ---- UTILS ----
def _densify_pair(a, b, max_step_deg):
    """Linear densification in lon/lat between two points a=(lon,lat), b=(lon,lat).
    Returns a list of coordinates starting at a and *including* b.
    """
    ax, ay = a; bx, by = b
    dx = abs(bx - ax); dy = abs(by - ay)
    steps = max(1, ceil(max(dx, dy) / max_step_deg))
    if steps == 1:
        return [a, b]
    out = []
    for t in range(steps):
        s = t / steps
        out.append((ax + s*(bx-ax), ay + s*(by-ay)))
    out.append((bx, by))  # ensure exact end
    return out

def _segmentize_linestring(ls, max_step_deg):
    """Yield consecutive (a,b) coordinate pairs whose max component step <= max_step_deg."""
    coords = list(ls.coords)
    if len(coords) < 2:
        return
    prev = coords[0]
    for k in range(1, len(coords)):
        chunk = _densify_pair(prev, coords[k], max_step_deg)
        # chunk includes prev and coords[k]; emit consecutive pairs
        for i in range(len(chunk)-1):
            a = chunk[i]; b = chunk[i+1]
            if a != b:
                yield a, b
        prev = coords[k]

def _iter_lines(g):
    """Iterate LineStrings from a geometry that may be LineString/MultiLineString/Collection."""
    if g.is_empty:
        return
    if g.geom_type == "LineString":
        yield g
    elif g.geom_type == "MultiLineString":
        for sub in g.geoms:
            if not sub.is_empty:
                yield sub
    else:
        # GeometryCollection or other: recurse into parts if present
        if hasattr(g, "geoms"):
            for sub in g.geoms:
                yield from _iter_lines(sub)

# ---- MAIN ----
def create_borders():
    # 1) Load countries as lon/lat (EPSG:4326)
    gdf = gpd.read_file(GPKG_PATH, layer=LAYER)
    if gdf.crs is None or (gdf.crs.to_epsg() or 4326) != 4326:
        gdf = gdf.to_crs(4326)

    gdf = gdf[[ID_FIELD, "geometry"]].reset_index(drop=True)

    # (Optional) light precision snap to reduce micro-slivers (uncomment if needed)
    # import shapely
    # gdf["geometry"] = shapely.set_precision(gdf.geometry.values, 1e-9)

    # 2) BBox shortlist with sjoin, then exact touches filter
    #   We do i<j to keep each neighbor pair once.
    short = gpd.sjoin(
        gdf[["geometry"]],
        gdf[["geometry"]],
        how="inner",
        predicate="intersects"
    )
    
    #print(short.columns)
    #return
    short = short[short.index < short.index_right]

    records = []

    # 3) For each candidate pair, confirm touches and extract shared boundary
    for i, j in zip(short.index.to_list(), short.index_right.to_list()):
        geom_i = gdf.geometry.iat[i]
        geom_j = gdf.geometry.iat[j]

        # Fast exact predicate
        try:
            if not geom_i.touches(geom_j):
                continue
        except GEOSException:
            continue

        # Intersection gives the shared line(s)
        inter = geom_i.intersection(geom_j)
        if inter.is_empty:
            continue

        id_i = int(gdf.iat[i, gdf.columns.get_loc(ID_FIELD)])
        id_j = int(gdf.iat[j, gdf.columns.get_loc(ID_FIELD)])
        id_a, id_b = (id_i, id_j) if id_i < id_j else (id_j, id_i)

        # 4) Segmentize each shared line and emit short segments
        for line in _iter_lines(inter):
            if line.is_empty or line.length == 0:
                continue
            for a, b in _segmentize_linestring(line, MAX_STEP_DEG):
                records.append({
                    "id_a": id_a,
                    "id_b": id_b,
                    "ax": float(a[0]), "ay": float(a[1]),
                    "bx": float(b[0]), "by": float(b[1]),
                    "geometry": LineString([a, b]),
                })

    if not records:
        raise RuntimeError("No shared borders were created. Check input geometries/CRS.")

    borders = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)

    # Dedup exact repeats (can happen with multipolygon boundaries)
    borders = borders.drop_duplicates(subset=["id_a", "id_b", "ax", "ay", "bx", "by"]).reset_index(drop=True)

    # 5) Save as FlatGeobuf (fast + spatial index)
    borders.to_file(OUT_FGB, driver="FlatGeobuf")
    print(f"Saved {len(borders):,} segments to {OUT_FGB}")



start = time.time()
create_borders()
end = time.time()
print(end - start)