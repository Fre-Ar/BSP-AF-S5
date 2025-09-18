# pip install geopandas pyogrio shapely
import geopandas as gpd
import shapely
from shapely.ops import unary_union

# 1) Load your canonical countries (land + oceans) into a GPKG
countries = gpd.read_file("python/geodata/world_bank_geodata.gpkg", layer="countries")  # has 'iso','name'
countries = countries.to_crs(4326)

# 2) Build shared borders with left/right iso (neighbor intersections)
borders = []
for i, a in countries.iterrows():
    # spatially join neighbors once for efficiency in real code
    neigh = countries[countries.geometry.touches(a.geometry)]
    for j, b in neigh.iterrows():
        if a.id >= b.id:  # avoid duplicates (order pair)
            continue
        inter = a.geometry.intersection(b.geometry)
        if inter.is_empty:
            continue
        inter = shapely.get_geometry(inter)  # normalize
        for line in shapely.geometry.GeometryCollection([inter]).geoms:
            if line.length > 0:
                borders.append({"left_id": a.id, "right_id": b.id, "geometry": line})
borders = gpd.GeoDataFrame(borders, geometry="geometry", crs=4326)

# Optional: segmentize long arcs so no segment spans huge distances
borders["geometry"] = borders.geometry.segmentize(0.25)  # densify every ~0.25°

# 3) Save runtime layers
borders.to_file("borders.fgb")              # FlatGeobuf
countries.to_parquet("countries.parquet")   # GeoParquet (via GeoPandas/pyarrow)
