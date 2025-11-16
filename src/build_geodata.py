# src/build_geodata.py

import multiprocessing as mp
import time
import os

from geodata.ecoc.adjacency import load_layer, build_adjacency
from geodata.ecoc.ecoc import gen_ecoc
from geodata.preprocess_borders import create_borders
from geodata.sampler.labeling import make_dataset_parallel


from utils.utils_geo import COUNTRIES_ECOC_PATH, ADJACENCY_JSON_PATH, \
    COUNTRIES_LAYER, ID_FIELD, GPKG_PATH, ECOC_BITS, SEED, BORDERS_FGB_PATH, FOLDER_PATH
from utils.utils import write_json, human_int

# -----------------------------
# Adjacency Graph
# -----------------------------

def create_adjacency_graph():
    gdf, id_field = load_layer(GPKG_PATH, COUNTRIES_LAYER, ID_FIELD)
    adj = build_adjacency(gdf, id_field=id_field)
  
    write_json(ADJACENCY_JSON_PATH, adj, name="adjacency")

# -----------------------------
# ECOC
# -----------------------------

def create_ecoc():
    gen_ecoc(ADJACENCY_JSON_PATH,
             COUNTRIES_ECOC_PATH,
             ECOC_BITS,
             SEED)
    
# -----------------------------
# Borders Preprocessing (FGB file creation)
# -----------------------------

def preprocess_borders():
    create_borders(GPKG_PATH, BORDERS_FGB_PATH, COUNTRIES_LAYER, ID_FIELD)

# -----------------------------
# Parquet File Creation
# -----------------------------

def create_training_data():
        # safety for multiprocessing
    mp.set_start_method("spawn", force=True)

    t0 = time.perf_counter()
    n = 1_000_000
    path = make_dataset_parallel(
        n_total=n,
        out_path=os.path.join(FOLDER_PATH, f"parquet/training_{human_int(n)}_.parquet"),
        mixture=(0.70, 0.30),
        shards_per_total=32,
        max_workers=None,
        seed=None,
        knn_k=128,
        knn_expand=256,
        expand_rel=1.05,
        reliable=True
    )
    dt = time.perf_counter() - t0
    print(f"Total time Elapsed: {dt:.3f}s")

if __name__ == "__main__":
    pass
