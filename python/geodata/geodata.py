import numpy as np
import geopandas as gpd
from shapely.strtree import STRtree
from shapely.geometry import Point

R_EARTH_KM = 6371.0088  # mean Earth radius

def lonlat_to_unitvec(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    clat = np.cos(lat)
    return np.array([clat*np.cos(lon), clat*np.sin(lon), np.sin(lat)])

def greatcircle_point_segment_dist_km(p_lon, p_lat, a_lon, a_lat, b_lon, b_lat):
    p = lonlat_to_unitvec(p_lon, p_lat)
    a = lonlat_to_unitvec(a_lon, a_lat)
    b = lonlat_to_unitvec(b_lon, b_lat)

    n = np.cross(a, b); n_norm = np.linalg.norm(n)
    if n_norm == 0:  # degenerate segment
        return R_EARTH_KM * np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
    n /= n_norm

    # Foot of perpendicular from p to great circle (a,b)
    c = np.cross(n, p)
    c = np.cross(c, n)
    c /= np.linalg.norm(c)

    # Check if c lies on short arc a-b; otherwise clamp to endpoints
    ab = np.arccos(np.clip(np.dot(a, b), -1.0, 1.0))
    ac = np.arccos(np.clip(np.dot(a, c), -1.0, 1.0))
    cb = np.arccos(np.clip(np.dot(c, b), -1.0, 1.0))

    if abs((ac + cb) - ab) < 1e-10:
        theta = np.arccos(np.clip(np.dot(p, c), -1.0, 1.0))
    else:
        theta = min(np.arccos(np.clip(np.dot(p, a), -1.0, 1.0)),
                    np.arccos(np.clip(np.dot(p, b), -1.0, 1.0)))
    return R_EARTH_KM * theta

class Sampler:
    def __init__(self, borders_fgb, countries_parquet_or_fgb):
        # Borders as small segments (with left/right_iso)
        self.borders = gpd.read_file(borders_fgb)
        # Precompute quick segment endpoints
        self._seg_end = np.array([
            (geom.coords[0][0], geom.coords[0][1], geom.coords[-1][0], geom.coords[-1][1])
            for geom in self.borders.geometry
        ])
        self._left = self.borders["left_iso"].to_numpy()
        self._right = self.borders["right_iso"].to_numpy()
        self._border_tree = STRtree(self.borders.geometry)

        # Countries polygons
        self.countries = gpd.read_file(countries_parquet_or_fgb)
        self._poly_tree = STRtree(self.countries.geometry)

    def country_at(self, lon, lat):
        pt = Point(lon, lat)
        # shortlist by bbox
        cand = self._poly_tree.query(pt)
        for poly in cand:
            idx = self.countries.index[self.countries.geometry == poly][0]
            if poly.contains(pt):
                return self.countries.loc[idx, "iso"]
        return None  # should not happen if oceans included

    def nearest_border(self, lon, lat):
        pt = Point(lon, lat)
        # rough 1-degree box for shortlist
        cand = self._border_tree.query(pt.buffer(1.0))
        if not cand:
            cand = self._border_tree.query(pt.buffer(5.0))
        best_d = 1e18
        best_idx = None
        for geom in cand:
            idx = self.borders.index[self.borders.geometry == geom][0]
            a_lon, a_lat, b_lon, b_lat = self._seg_end[idx]
            d = greatcircle_point_segment_dist_km(lon, lat, a_lon, a_lat, b_lon, b_lat)
            if d < best_d:
                best_d, best_idx = d, idx
        left = self._left[best_idx]; right = self._right[best_idx]
        return best_d, left, right, best_idx

    def sample(self, lon, lat):
        c1 = self.country_at(lon, lat)
        dist_km, left, right, _ = self.nearest_border(lon, lat)
        # Pick c2 as the "other" side of the nearest border
        c2 = right if c1 == left else left if c1 == right else right
        return dist_km, c1, c2


class Tester:
    def __init__(self, borders_fgb, countries_parquet_or_fgb):
        # Borders as small segments (with left/right_iso)
        self.borders = gpd.read_file(borders_fgb)
        # Precompute quick segment endpoints
        self._seg_end = np.array([
            (geom.coords[0][0], geom.coords[0][1], geom.coords[-1][0], geom.coords[-1][1])
            for geom in self.borders.geometry
        ])
        self._left = self.borders["left_iso"].to_numpy()
        self._right = self.borders["right_iso"].to_numpy()
        self._border_tree = STRtree(self.borders.geometry)

        # Countries polygons
        self.countries = gpd.read_file(countries_parquet_or_fgb)
        self._poly_tree = STRtree(self.countries.geometry)

    def country_at(self, lon, lat):
        pt = Point(lon, lat)
        # shortlist by bbox
        cand = self._poly_tree.query(pt)
        for poly in cand:
            idx = self.countries.index[self.countries.geometry == poly][0]
            if poly.contains(pt):
                return self.countries.loc[idx, "iso"]
        return None  # should not happen if oceans included

    def nearest_border(self, lon, lat):
        pt = Point(lon, lat)
        # rough 1-degree box for shortlist
        cand = self._border_tree.query(pt.buffer(1.0))
        if not cand:
            cand = self._border_tree.query(pt.buffer(5.0))
        best_d = 1e18
        best_idx = None
        for geom in cand:
            idx = self.borders.index[self.borders.geometry == geom][0]
            a_lon, a_lat, b_lon, b_lat = self._seg_end[idx]
            d = greatcircle_point_segment_dist_km(lon, lat, a_lon, a_lat, b_lon, b_lat)
            if d < best_d:
                best_d, best_idx = d, idx
        left = self._left[best_idx]; right = self._right[best_idx]
        return best_d, left, right, best_idx

    def sample(self, lon, lat):
        c1 = self.country_at(lon, lat)
        dist_km, left, right, _ = self.nearest_border(lon, lat)
        # Pick c2 as the "other" side of the nearest border
        c2 = right if c1 == left else left if c1 == right else right
        return dist_km, c1, c2