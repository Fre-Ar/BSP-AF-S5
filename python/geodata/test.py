import numpy as np, pandas as pd, geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree
from pyproj import Geod, Transformer

GEOD = Geod(ellps="WGS84")
R_EARTH_KM = 6371.0088
GPKG = "python/geodata/world_bank_geodata.gpkg"  # your file
LAYER = "countries"               # adjust if needed
gdf = gpd.read_file(GPKG, layer=LAYER).to_crs(4326)

# fix invalid
bad = ~gdf.geometry.is_valid
if bad.any():
    gdf.loc[bad, "geometry"] = gdf.loc[bad, "geometry"].buffer(0)

# random points on the sphere (Fibonacci)
def fib_sphere(n):
    k = np.arange(n) + 0.5
    lon = 360*(k/ n)  # 0..360
    lat = np.degrees(np.arcsin(1 - 2*k/n))
    lon = (lon + 180) % 360 - 180
    return np.c_[lon, lat]

pts = fib_sphere(100)
pts_g = gpd.GeoSeries([Point(lon, lat) for lon,lat in pts], crs=4326)

# point-in-polygon via STRtree (fast shortlist)
tree = STRtree(gdf.geometry.values)
def count_hits(pt):
    cands = tree.query(pt)
    return sum(poly for poly in cands)

hits = np.array([count_hits(pt) for pt in pts_g])
print(hits)
#print("Coverage OK? (all == 1):", (hits==1).mean(), "fraction")
#print("Overlaps:", (hits>1).sum(), "Uncovered:", (hits==0).sum())
