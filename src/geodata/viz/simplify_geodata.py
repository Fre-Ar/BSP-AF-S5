import os
import geopandas as gpd
from src.utils.utils_geo import GPKG_PATH, COUNTRIES_LAYER

def gpkg_to_geojson():
    g = gpd.read_file(GPKG_PATH, layer=COUNTRIES_LAYER)
    g.to_file("python/geodata/data/simp/countries.json", driver="GeoJSON")

def geojson_to_gpkg():
    g = gpd.read_file("python/geodata/data/simp/out_countries.json")
    if g.crs is None:
        g = g.set_crs(4326)
    g.to_file("python/geodata/data/countries_simp_01p.gpkg", layer=COUNTRIES_LAYER, driver="GPKG", index=False)

import os
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"  # 0 = no limit
geojson_to_gpkg()