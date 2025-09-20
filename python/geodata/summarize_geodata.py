import numpy as np
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.strtree import STRtree
import time

GPKG_PATH = "python/geodata/world_bank_geodata.gpkg"
LAYER = "countries"
ID = "id"  # field with unique country identifier
OUT_FGB = "python/geodata/borders.fgb"

SEGMENTIZE_DEG = 0.25 # How finely to densify borders before splitting into segments (degrees)
R_EARTH_KM = 6371.0088  # mean Earth radius

def xyz_to_lonlat(x, y, z):
        # assuming (x,y,z) already normalized to unit sphere
        lon = np.degrees(np.arctan2(y, x))  # [-180, 180)
        lat = np.degrees(np.arcsin(np.clip(z, -1.0, 1.0)))
        return float(lon), float(lat)
    
def lonlat_to_unitvec(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    clat = np.cos(lat)
    return np.array([clat*np.cos(lon), clat*np.sin(lon), np.sin(lat)], dtype=float)

def greatcircle_point_segment_dist_km(p_lon, p_lat, a_lon, a_lat, b_lon, b_lat):
    """Spherical point-to-segment distance (km) on the unit sphere."""
    p = lonlat_to_unitvec(p_lon, p_lat)
    a = lonlat_to_unitvec(a_lon, a_lat)
    b = lonlat_to_unitvec(b_lon, b_lat)

    n = np.cross(a, b)
    n_norm = np.linalg.norm(n)
    if n_norm == 0.0:
        # Degenerate segment: use nearest endpoint
        return R_EARTH_KM * min(
            np.arccos(np.clip(np.dot(p, a), -1.0, 1.0)),
            np.arccos(np.clip(np.dot(p, b), -1.0, 1.0)),
        )
    n /= n_norm

    # Foot of perpendicular from p to GC(a,b)
    c = np.cross(n, p)
    c = np.cross(c, n)
    c /= np.linalg.norm(c)

    # Check if c lies on the short arc a->b; else clamp to endpoints
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
    
class CountryLocator:
    """
    country_id_at_xyz(x,y,z) -> id
    country_and_border_at_xyz(x,y,z) -> (id_country, distance_km_to_nearest_border, id_other_country)
    """
    def __init__(self, gpkg_path=GPKG_PATH, layer=LAYER, id_field=ID,
                 segmentize_deg=SEGMENTIZE_DEG):
        gdf = gpd.read_file(gpkg_path, layer=layer)
        # Ensure CRS is lon/lat WGS84
        if gdf.crs is None or int(gdf.crs.to_epsg() or 4326) != 4326:
            gdf = gdf.to_crs(4326)
        # Keep only geometry + id for speed
        self.id_field = id_field
        self.countries = gdf[[id_field, "geometry"]].copy().reset_index(drop=True)

        # Spatial index for polygons
        self._poly_geoms = list(self.countries.geometry.values)
        self._poly_tree = STRtree(self._poly_geoms)

        # --- Build shared borders (left_id/right_id) ---
        # Strategy: for each polygon, shortlist neighbors by bbox-touch, compute line intersection,
        # keep non-empty line(s). Segmentize to limit arc length for better spherical distance.
        borders_records = []
        # Reuse a polygon STRtree for neighbor discovery
        for i, (id_i, geom_i) in enumerate(zip(self.countries[self.id_field], self._poly_geoms)):
            # bbox shortlist
            neigh_idxs = self._poly_tree.query(geom_i.boundary)
            for j in neigh_idxs:
                if j <= i:
                    continue
                id_j = int(self.countries.iloc[j][self.id_field])
                geom_j = self._poly_geoms[j]
                # Quick reject: if they don't touch, skip
                if not geom_i.touches(geom_j):
                    continue
                inter = geom_i.intersection(geom_j)
                if inter.is_empty:
                    continue
                # 'inter' can be MultiLineString/LineString
                for line in getattr(inter, "geoms", [inter]):
                    if line.length == 0:
                        continue
                    # Densify to ~segmentize_deg (in degrees) for better spherical approximation
                    try:
                        dense = line.segmentize(segmentize_deg)
                    except Exception:
                        dense = line  # Shapely <2.0 fallback; remove if not needed
                    coords = list(dense.coords)
                    # Emit consecutive segments as individual LineStrings with metadata
                    for k in range(len(coords) - 1):
                        a = coords[k]; b = coords[k+1]
                        if a == b: 
                            continue
                        borders_records.append({
                            "left_id": int(id_i),
                            "right_id": int(id_j),
                            "geometry": LineString([a, b]),
                            # cache endpoints for spherical distance
                            "_a_lon": float(a[0]), "_a_lat": float(a[1]),
                            "_b_lon": float(b[0]), "_b_lat": float(b[1]),
                        })

        self.borders = gpd.GeoDataFrame(borders_records, geometry="geometry", crs=4326)
        # Index for border segments
        self._seg_geoms = list(self.borders.geometry.values)
        self._seg_tree = STRtree(self._seg_geoms)
        # For builds that return geometries instead of indices
        self._seg_wkb_to_pos = {g.wkb: i for i, g in enumerate(self._seg_geoms)}
    
    def _poly_positions(self, geom_or_idxs):
        arr = np.asarray(geom_or_idxs)
        if arr.dtype.kind in ("i", "u"):
            return arr
        return np.array([self._poly_geoms.index(g) for g in geom_or_idxs], dtype=int)

    def _seg_positions(self, geom_or_idxs):
        arr = np.asarray(geom_or_idxs)
        if arr.dtype.kind in ("i", "u"):
            return arr
        return np.array([self._seg_wkb_to_pos[g.wkb] for g in geom_or_idxs], dtype=int)

    def country_id_at_xyz(self, x, y, z):
        lon, lat = xyz_to_lonlat(x, y, z)
        pt = Point(lon, lat)
        pos_candidates = self._poly_positions(self._poly_tree.query(pt))
        for pos in pos_candidates:
            geom = self._poly_geoms[pos]
            if geom.contains(pt):
                return int(self.countries.iloc[pos][self.id_field])
        return None  # should not happen if oceans are included

    def country_and_border_at_xyz(self, x, y, z):
        """
        Returns: (country_id, distance_km_to_nearest_international_border, other_country_id)
        Distance is always positive (unsigned).
        """
        lon, lat = xyz_to_lonlat(x, y, z)
        pt = Point(lon, lat)

        # 1) containing country
        cid = self.country_id_at_xyz(x, y, z)

        # 2) nearest international border (shortlist by small buffer in degrees)
        # Start with ~1° search; if nothing, widen.
        for buf_deg in (1.0, 3.0, 8.0):
            pos_segments = self._seg_positions(self._seg_tree.query(pt.buffer(buf_deg)))
            if len(pos_segments) > 0:
                break
        if len(pos_segments) == 0:
            # Highly unlikely, but fallback to all segments
            pos_segments = np.arange(len(self._seg_geoms))

        best_d = 1e18
        best_left = None
        best_right = None

        # Evaluate spherical distance to each candidate segment
        seg_df = self.borders  # local
        for pos in pos_segments:
            row = seg_df.iloc[pos]
            d_km = greatcircle_point_segment_dist_km(
                lon, lat, row["_a_lon"], row["_a_lat"], row["_b_lon"], row["_b_lat"]
            )
            if d_km < best_d:
                best_d = d_km
                best_left = int(row["left_id"])
                best_right = int(row["right_id"])

        # 3) other country on that nearest border
        if cid == best_left:
            other = best_right
        elif cid == best_right:
            other = best_left
        else:
            # If the nearest border doesn't bound the containing polygon (rare due to
            # numerical effects or tiny islands), pick the "right" side arbitrarily.
            other = best_right

        return int(cid), float(best_d), int(other)

def create_borders():
    # 1) Load countries layer (WGS84 lon/lat)
    gdf = gpd.read_file(GPKG_PATH, layer=LAYER)
    if gdf.crs is None or (gdf.crs.to_epsg() or 4326) != 4326:
        gdf = gdf.to_crs(4326)

    # Keep only id + geometry
    gdf = gdf[[ID, "geometry"]].reset_index(drop=True)

    # Optional topology cleanup to avoid micro-slivers (safe small precision)
    # Comment out if your data is already very clean.
    #try:
        #gdf["geometry"] = shapely.set_precision(gdf.geometry.values, 1e-9)
    #except Exception:
        #pass

    # 2) Spatial join to get touching neighbors (each pair once)
    # Using bbox predicate for speed, exact touch test later.
    # GeoPandas >=0.12 supports predicate="intersects"/"touches"
    idx_left = []
    idx_right = []

    # Fast bbox join to shortlist neighbors
    sjoin = gpd.sjoin(gdf[["geometry"]], gdf[["geometry"]], how="inner", predicate="intersects")
    # Keep only i<j to avoid duplicates and self-joins
    sjoin = sjoin[sjoin.index_left < sjoin.index_right]

    # 3) For each candidate pair, keep only true touching pairs and extract shared boundary
    records = []
    for i, j in zip(sjoin.index_left.values.tolist(), sjoin.index_right.values.tolist()):
        geom_i = gdf.geometry.iloc[i]
        geom_j = gdf.geometry.iloc[j]
        if not geom_i.touches(geom_j):
            continue

        inter = geom_i.intersection(geom_j)
        if inter.is_empty:
            continue

        # 'inter' may be MultiLineString or LineString
        lines = getattr(inter, "geoms", [inter])
        for line in lines:
            if line.is_empty or line.length == 0:
                continue
            # Densify line (Shapely 2.0); fallback to original if not available
            try:
                dense = line.segmentize(SEGMENTIZE_DEG)
            except Exception:
                dense = line
            coords = list(dense.coords)
            # Emit consecutive short segments
            for k in range(len(coords) - 1):
                a = coords[k]
                b = coords[k + 1]
                if a == b:
                    continue
                # Normalize id ordering so each border is consistently (min,max)
                id_i = int(gdf.iloc[i][ID])
                id_j = int(gdf.iloc[j][ID])
                id_a, id_b = (id_i, id_j) if id_i < id_j else (id_j, id_i)
                records.append({
                    "id_a": id_a,
                    "id_b": id_b,
                    "ax": float(a[0]), "ay": float(a[1]),
                    "bx": float(b[0]), "by": float(b[1]),
                    "geometry": LineString([a, b]),
                })

    borders = gpd.GeoDataFrame(records, geometry="geometry", crs=4326)

    # Optional: drop duplicates (can happen with multipolygons)
    borders = borders.drop_duplicates(subset=["id_a", "id_b", "ax", "ay", "bx", "by"])

    # 4) Save as FlatGeobuf (fast, has spatial index)
    borders.to_file(OUT_FGB, driver="FlatGeobuf")
    print(f"Saved {len(borders):,} segments to {OUT_FGB}")


def main():
    start = time.time()
    locator = CountryLocator()
    result = locator.country_and_border_at_xyz(1.0, 0.0, 0.0)  # point on equator at lon=0°
    print(result)
    end = time.time()
    print(end - start)
    
#main()
#print("done")
