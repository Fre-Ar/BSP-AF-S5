# src/geodata/ecoc/adjacency.py
import json
from collections import defaultdict
import geopandas as gpd, numpy as np
from shapely.validation import make_valid
from shapely.strtree import STRtree
from typing import Optional

def load_layer(path: str, layer: Optional[str], id_field: str):
    '''
    Loads a polygon layer, ensures one geometry per id, and fixes invalid shapes.
    
    Parameters
    ----------
    path : str
        Path to the vector datasource (e.g. GeoPackage file).
    layer : str or None
        Name of the layer to read from `path`.
    id_field : str
        Name of the column containing the unique identifier for each polygon.

    Returns
    -------
    dissolved : GeoDataFrame
        GeoDataFrame with one row per `id_field` value. Invalid geometries are
        repaired, empties dropped, and multipart geometries dissolved so each id
        has a single (multi)polygon.
    id_field : str
        The identifier field name (returned for convenience).
    '''
    # Read layer
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

def ensure_projected_for_length(gdf: gpd.GeoDataFrame):
    '''
    Ensures the gdf is in metric CRS, not in a degree-based projection.
    '''
    # Ensure meters for length thresholds (EPSG:8857 Equal Earth)
    if gdf.crs is None or getattr(gdf.crs, "is_geographic", False):
        return gdf.to_crs(8857)
    return gdf

def build_adjacency(gdf: gpd.GeoDataFrame, id_field: str,
                    min_shared_m: float = 0.0, use_intersects: bool = False):
    '''
    Builds the adjacency graph in the form of a {id : list of neighbour ids} dictionary.
    
    Notes:
    - Two polygons are adjacent if their boundaries *touch* (topological touch).
      If `use_intersects=True`, adjacency is widened to any intersection (including
      overlaps). For most country borders, `touches` is the right notion.
    
    Parameters
    ----------
    gdf : GeoDataFrame
        GeoDataFrame with polygon geometries and an identifier column.
    id_field : str
        Name of the identifier column in `gdf`.
    min_shared_m : float, optional
        Minimum shared boundary length in meters for two polygons to be considered
        neighbors. If 0.0, any topological contact is accepted (subject to the
        chosen predicate).
    use_intersects : bool, optional
        If False, two polygons are adjacent only when their boundaries *touch*
        (`g.touches(gj)`). If True, adjacency is widened to any intersection
        (`g.intersects(gj)`), including overlaps.

    Returns
    -------
    adjacency : dict
        Dictionary mapping `id_field` values to a sorted list of neighbor ids.
    '''
    # gets ids and polygons
    ids = gdf[id_field].tolist()
    geoms = gdf.geometry.tolist()

    # Spatial index
    tree = STRtree(geoms)
    # Map geometry object identity -> index
    geom_index = {id(geom): i for i, geom in enumerate(geoms)}

    # For length thresholds, use a metric CRS
    gdf_m = ensure_projected_for_length(gdf) if min_shared_m > 0 else gdf
    geoms_m = gdf_m.geometry.tolist()

    neighbors = defaultdict(set)

    # Decide how to query candidates depending on Shapely capabilities
    def candidate_indices(g):
        # Try query with predicate first
        try:
            cand = tree.query(g, predicate="intersects")
        except TypeError:
            # Very old Shapely: no predicate kwarg
            cand = tree.query(g)  # broad phase (geoms or indices), filter later

        if len(cand) == 0:
            return []

        first = cand[0]
        # If we got integer indices 
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

            # Narrow-phase predicate,
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

    adjacency = {k: sorted(v) for k, v in neighbors.items()}
    return adjacency

def load_graph(path: str):
    """
    Load an adjacency JSON dict and convert it to a compact, undirected edge list.

    Parameters
    ----------
    path : str
        Path to a JSON file with structure
        ``{ node_id: [neighbor_id1, neighbor_id2, ...], ... }``.
        Node ids and neighbor ids may be strings or integers.

    Returns
    -------
    ids : list
        List of canonical node ids (ints when possible, else original strings).
    id2idx : dict
        Mapping from canonical id -> contiguous index [0..N-1].
    idx2id : dict
        Inverse mapping from index -> canonical id.
    edges_array : ndarray
        (M, 2) int32 array of undirected edges over indices, with i < j.
    """
    with open(path, "r") as f:
        raw = json.load(f)
    
    def normalize_id(id):
        try:
            cid = int(id)
        except (ValueError, TypeError):
            cid = id
        return id

    # Collect all canonical ids from keys (normalize numeric strings -> ints)
    ids = []
    for k in raw.keys():
        ids.append(normalize_id(k))
    
    # Deterministic ordering: all ints (numeric sort), then all strings (lex sort)
    ids = sorted(ids, key=lambda x: (isinstance(x, str), x))

    # Map id -> idx and back
    id2idx = {cid: i for i, cid in enumerate(ids)}
    idx2id = {i: cid for cid, i in id2idx.items()}

    # Build undirected edge set on indices (i < j)
    edges = set()
    
    for k, nbrs in raw.items():
        a = normalize_id(k)
        if a not in id2idx:  # skip unknown ids (shouldn't happen); skip defensively
                continue
        ia = id2idx[a]
        
        for b in nbrs:
            b = normalize_id(b)
            if b not in id2idx:  # skip unknown ids (shouldn't happen)
                continue
            ib = id2idx[b]
            
            if ia == ib:
                # Skip self-loops
                continue
            
            i, j = (ia, ib) if ia < ib else (ib, ia)
            edges.add((i, j))
            
    edges = sorted(edges)
    edges_array = np.array(edges, dtype=np.int32) if edges else np.empty((0, 2), dtype=np.int32)
    return ids, id2idx, idx2id, edges_array
