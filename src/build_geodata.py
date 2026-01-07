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
    """
    100% uniform:
        100k points takes ~46s
        1M   points takes ~220s
        10M  points takes ~1600s or ~26 min
    100% border:
        100k points takes ~46s
        1M   points takes ~148s
        10M  points takes ~1160s or ~19 min
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
    
# -----------------------------
# Parquet Bundling
# -----------------------------

def bundle_training_data(
    source_subdir: str = "training_2M", 
    target_subdir: str = "training", 
    bundle_factor: int = 5,
    final_points_per_file: int = 10_000_000
):
    """
    Concatenates smaller parquet files into larger ones.
    
    Default settings (100 -> 20 files):
    - source: src/geodata/parquet/training_2M
    - target: src/geodata/parquet/training
    - bundle_factor: 5 (merges 5 files into 1)
    """
    source_dir = os.path.join(TRAINING_DATA_PATH, source_subdir)
    target_dir = os.path.join(TRAINING_DATA_PATH, target_subdir)
    
    os.makedirs(target_dir, exist_ok=True)

    # 1. Gather sorted files
    files = sorted(glob.glob(os.path.join(source_dir, "*.parquet")))
    if not files:
        print(f"No parquet files found in {source_dir}")
        return

    total_files = len(files)
    num_bundles = math.ceil(total_files / bundle_factor)
    h_int = human_int(final_points_per_file)

    print(f"Found {total_files} files. Bundling into ~{num_bundles} files (factor {bundle_factor})...")

    # 2. Batch process
    for i in range(num_bundles):
        start = i * bundle_factor
        end = min(start + bundle_factor, total_files)
        batch_paths = files[start:end]

        out_name = f"training_{h_int}_{i:02d}.parquet"
        out_path = os.path.join(target_dir, out_name)

        print(f"[{i+1}/{num_bundles}] Merging {len(batch_paths)} files -> {out_name}...")
        
        # Reuse existing memory-efficient concat utility
        _concat_parquet_shards(
            shard_paths=batch_paths,
            out_path=out_path
        )

    print(f"Done. New files located in: {target_dir}")


if __name__ == "__main__":
    #pass
    create_training_data()
