# src/main.py
import os
import time

from nirs.viz.compare_data_viz import visualize_model
from nirs.viz.rasterizer import raster
from nirs.training import train_and_eval
from utils.utils_geo import TRAINING_DATA_PATH
from config import *

def train():
    """
    On MPS:
        1mb model:
            20 epochs -> 150s (99s if plugged in)
        4mb model:
            20 epochs -> 240s
    """
    if TRAINING_POINTS == 1_000_000:
        SIZE = "1M"
    elif TRAINING_POINTS == 10_000_000:
        SIZE = "10M"
    TRAIN_PATH = os.path.join(TRAINING_DATA_PATH, f"eval_uniform_{SIZE}.parquet")
    EVAL_PATH = os.path.join(TRAINING_DATA_PATH, f"eval_uniform_1M.parquet")
    #PATH = os.path.join(TRAINING_DATA_PATH, f"log_dataset_{SIZE}.parquet")
    #PATH = os.path.join(TRAINING_DATA_PATH, f"training_{SIZE}.parquet")
    
    t0 = time.perf_counter()
    train_and_eval(
        TRAIN_PATH,
        model_cfg=MODEL_CONFIG,
        eval_set_path=EVAL_PATH,
        epochs=EPOCHS,
        batch_size = 8192,
        traning_size = TRAINING_POINTS,
        #lr=3e-4,
        )
    dt = time.perf_counter() - t0
    print(f"Total training time Elapsed: {dt:.3f}s")

def viz(pred: bool = False):  
    visualize_model(
        parquet_path=os.path.join(TRAINING_DATA_PATH, "log_dataset_1M.parquet"),
        #parquet_path=os.path.join(TRAINING_DATA_PATH, "eval_uniform_1M.parquet"),
        #parquet_path=os.path.join(TRAINING_DATA_PATH, "eval_border_1M.parquet"),
    
        checkpoint_path=MODEL_PATH,
        model_cfg=MODEL_CONFIG,
        
        sample=1_000_000,
        predictions_only=pred,
        show_plots=False,
        overrides={180: "#000000"} #australia becomes black
        )

def img():
    t0 = time.perf_counter()

    raster(
        model_cfg=MODEL_CONFIG,
        checkpoint_path=MODEL_PATH,
        render = "c1",
        area="globe")
    
    dt = time.perf_counter() - t0
    print(f"Total rasterization time Elapsed: {dt:.3f}s")

if __name__ == "__main__":
    #pass
    #train()
    viz(True)
    #img()
