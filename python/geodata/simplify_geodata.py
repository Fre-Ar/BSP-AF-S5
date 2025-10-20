import json, subprocess, tempfile, os, math
import geopandas as gpd


def count_edges(geojson_path):
    # Use mapshaper to dump topology stats (arcs length = edge count proxy)
    cmd = ["mapshaper", geojson_path, "-topology", "-info", "-quiet"]
    out = subprocess.check_output(cmd, text=True)
    # Mapshaper prints arc count like "arcs: N"; parse N:
    for line in out.splitlines():
        line=line.strip().lower()
        if line.startswith("arcs:"):
            return int(line.split(":")[1].strip())
    return None

def simplify_to_budget(in_gpkg, in_layer, out_gpkg, target_edges=1000):
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src.geojson")
        # export from gpkg to geojson in equal-area
        subprocess.check_call([
            "mapshaper", in_gpkg, f"layers={in_layer}",
            "-proj", "equalearth",
            "-o", src
        ])

        lo, hi = 0.1, 50.0  # % range to search (lo=strong simplification)
        best = None
        for _ in range(12):  # ~binary search
            mid = (lo + hi) / 2.0
            test = os.path.join(td, f"test_{mid:.2f}.geojson")
            subprocess.check_call([
                "mapshaper", src, "-topology",
                "-simplify", f"visvalingam keep-shapes {mid}%",
                "-o", test
            ])
            edges = count_edges(test)
            # print(mid, edges)
            if edges is None:
                break
            best = (mid, test, edges)
            if edges > target_edges:
                # too many edges -> simplify more (increase %)
                lo = mid
            else:
                # too few/ok -> simplify less (decrease %)
                hi = mid

        # write best as gpkg
        _, test_geojson, _ = best
        subprocess.check_call([
            "mapshaper", test_geojson, "-o",
            out_gpkg, "format=gpkg", "layer="+in_layer+"_simpl"
        ])



# usage
# simplify_to_budget("python/geodata/world_bank_geodata.gpkg", "countries",
#                    "python/geodata/world_bank_geodata_simpl.gpkg", target_edges=1000)

def gpkg_to_geojson():
    g = gpd.read_file("python/geodata/world_bank_geodata.gpkg", layer="countries")
    g.to_file("python/geodata/countries.json", driver="GeoJSON")

def geojson_to_gpkg():
    g = gpd.read_file("python/geodata/out_countries.json")
    if g.crs is None:
        g = g.set_crs(4326)
    g.to_file("python/geodata/countries_simp_01p.gpkg", layer="countries", driver="GPKG", index=False)

import os
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"  # 0 = no limit
geojson_to_gpkg()