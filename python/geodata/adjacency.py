import json
import argparse
from collections import defaultdict
import geopandas as gpd, numpy as np
from shapely.validation import make_valid
from shapely.strtree import STRtree
from shapely import touches
from tqdm import tqdm

def load_layer(path, layer=None, id_field="id"):
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if id_field not in gdf.columns:
        raise ValueError(f"ID field '{id_field}' not found. Available: {list(gdf.columns)}")
    # Keep only id and geometry
    gdf = gdf[[id_field, gdf.geometry.name]].copy()
    # Fixing invalid geometries
    gdf[gdf.geometry.name] = gdf.geometry.apply(make_valid)
    # Drop empties
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    # Dissolve by id to merge multipart territories into a single geometry per id
    dissolved = gdf.dissolve(by=id_field, as_index=False)
    dissolved = dissolved.reset_index(drop=True)
    return dissolved, id_field

def ensure_projected_for_length(gdf):
    # Ensure meters for length thresholds (EPSG:8857 Equal Earth)
    if gdf.crs is None or getattr(gdf.crs, "is_geographic", False):
        return gdf.to_crs(8857)
    return gdf

def build_adjacency(gdf, id_field="id", min_shared_m=0.0, use_intersects=False):
    """
    Returns dict[id] -> sorted list of neighbor ids (border touch by default).
    Works across Shapely versions (with/without `predicate`, `return_geometries`).
    """
    ids = gdf[id_field].tolist()
    geoms = gdf.geometry.tolist()

    # Spatial index
    tree = STRtree(geoms)
    # Map geometry object identity -> index (for Shapely versions that return geoms)
    geom_index = {id(geom): i for i, geom in enumerate(geoms)}

    # For length thresholds, use a metric CRS
    gdf_m = ensure_projected_for_length(gdf) if min_shared_m > 0 else gdf
    geoms_m = gdf_m.geometry.tolist()

    neighbors = defaultdict(set)

    # Decide how to query candidates depending on Shapely capabilities
    def candidate_indices(g):
        # Try query with predicate first (newer Shapely)
        try:
            cand = tree.query(g, predicate="intersects")
        except TypeError:
            # Very old Shapely: no predicate kwarg
            cand = tree.query(g)  # broad phase (geoms or indices), filter later

        if len(cand) == 0:
            return []

        first = cand[0]
        # If we got integer indices (Shapely >=2 typically returns ndarray of ints)
        if isinstance(first, (int, np.integer)):
            return cand
        # Otherwise, we got geometries: map back to indices
        return [geom_index[id(gg)] for gg in cand]

    for i, g in enumerate(geoms):
        cand_idx = candidate_indices(g)
        gi_m = geoms_m[i]

        for j in cand_idx:
            if j == i:
                continue

            gj = geoms[j]

            # Narrow-phase predicate, written for cross-version compatibility
            border_ok = gj.intersects(g) if use_intersects else g.touches(gj)
            if not border_ok:
                continue

            # Optional minimum shared-border length threshold (in meters)
            if min_shared_m > 0:
                gj_m = geoms_m[j]
                shared_len = gi_m.boundary.intersection(gj_m.boundary).length
                if shared_len < min_shared_m:
                    continue

            neighbors[ids[i]].add(ids[j])
        print(i, "->", len(neighbors[ids[i]]), "neighbors")
        

    return {k: sorted(v) for k, v in neighbors.items()}



def main():
    PATH = "python/geodata/world_bank_geodata.gpkg"
    LAYER = "countries"
    ID = "id"
    gdf, id_field = load_layer(PATH, LAYER, ID)
    adj = build_adjacency(gdf, id_field=id_field)

   
    OUT = "python/geodata/geodata_adjacency.json"
    if OUT:
        with open(OUT, "w", encoding="utf-8") as f:
            json.dump(adj, f, ensure_ascii=False, indent=2)
        print(f"Wrote adjacency JSON to {OUT}")
    else:
        print(json.dumps(adj, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
