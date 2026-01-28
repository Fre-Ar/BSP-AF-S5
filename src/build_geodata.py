# src/build_geodata.py

import multiprocessing as mp
import time
import os
import glob
import math

from geodata.ecoc.adjacency import load_layer, build_adjacency
from geodata.ecoc.ecoc import gen_ecoc
from geodata.preprocess_borders import create_borders
from geodata.sampler.labeling import make_batch_datasets


from utils.utils_geo import COUNTRIES_ECOC_PATH, ADJACENCY_JSON_PATH, \
    COUNTRIES_LAYER, ID_FIELD, GPKG_PATH, ECOC_BITS, SEED, BORDERS_FGB_PATH, TRAINING_DATA_PATH
from utils.utils import write_json, human_int, _concat_parquet_shards

# -----------------------------
# Adjacency Graph
# -----------------------------

def create_adjacency_graph():
    """
    Creates an adjacency graph from the specified countries layer for ECOC generation.
    The graph is saved as a JSON file at the specified path.
    """
    gdf, id_field = load_layer(GPKG_PATH, COUNTRIES_LAYER, ID_FIELD)
    adj = build_adjacency(gdf, id_field=id_field)
  
    write_json(ADJACENCY_JSON_PATH, adj, name="adjacency")

# -----------------------------
# ECOC
# -----------------------------

def create_ecoc():
    """
    Generates Error-Correcting Output Codes (ECOC) for the countries based on the adjacency graph.
    The ECOC are saved to the specified path.
    """
    gen_ecoc(ADJACENCY_JSON_PATH,
             COUNTRIES_ECOC_PATH,
             ECOC_BITS,
             SEED)
    
# -----------------------------
# Borders Preprocessing (FGB file creation)
# -----------------------------

def preprocess_borders():
    """
    Preprocesses country borders from the GPKG file and creates a FlatGeobuf (FGB) file.
    """
    create_borders(GPKG_PATH, BORDERS_FGB_PATH, COUNTRIES_LAYER, ID_FIELD)

# -----------------------------
# Parquet File Creation
# -----------------------------

def create_training_data():
    """
    Creates training data Parquet files with biased sampling.
    Generates multiple files with specified number of points and saves them to the training data path.
    """
    # safety for multiprocessing
    mp.set_start_method("spawn", force=True)

    t0 = time.perf_counter()
    n = 10_000_000
    num_files = 20

    def path_gen(i: int) -> str:
        return os.path.join(
            TRAINING_DATA_PATH, 
            "training_biased", 
            f"{human_int(n)}_{i:02d}.parquet"
        )
    paths = make_batch_datasets(
        n_total_per_file=n,
        num_files=num_files,
        path_generator=path_gen,
        mixture=(0.5, 0.5),
        shards_per_file=32, 
        max_workers=None,
        seed=None,
        knn_k=128,
        knn_expand=256,
        expand_rel=1.05
    )
    dt = time.perf_counter() - t0
    print(f"Total time Elapsed: {dt:.3f}s")
    
if __name__ == "__main__":
    pass
    #create_adjacency_graph()
    #create_ecoc()
    #preprocess_borders() 
    #create_training_data()
