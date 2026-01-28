# src/main.py
import time
import random
import os
import numpy as np
import torch

from nirs.viz.compare_data_viz import visualize_model
from nirs.viz.rasterizer import raster
from nirs.training import train_and_eval
from utils.utils_geo import SEED, TRAINING_DATA_PATH, BEST_LOGS_PATH, BEST_CHECKPOINT_PATH
from config import *

def seed_everything(seed: int = SEED):
    """
    Seeds all random number generators to ensure reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    # Torch (CPU + CUDA)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    
    # Torch (MPS - Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # Force deterministic algorithms (Optional: may slow down training)
    # torch.use_deterministic_algorithms(True) 
    
    print(f"[Info] Global seed set to: {seed}")

def train(lr = LR):
    """
    Trains and evaluates the NIR model using the specified training directory,
    model configuration, and evaluation dataset.
    """

    t0 = time.perf_counter()
    train_and_eval(
        train_dir=TRAIN_BIAS_DIR,
        model_cfg=MODEL_CONFIG,
        eval_set_path=EVAL_PATH,
        file_prefix="10M_",
        out_dir=BEST_CHECKPOINT_PATH,
        log_dir=BEST_LOGS_PATH,
        batch_size = 4096,
        traning_size = TRAINING_POINTS,
        lr=lr,
        weight_decay=WD,
        )
    dt = time.perf_counter() - t0
    print(f"Total training time Elapsed: {dt:.3f}s")

def viz(pred: bool = False):  
    """
    Visualizes model predictions against ground truth data.
    If `pred` is True, only model predictions are shown.
    """
    visualize_model(
        parquet_path=os.path.join(TRAINING_DATA_PATH, "eval_uniform_1M.parquet"),
    
        checkpoint_path=MODEL_PATH,
        model_cfg=MODEL_CONFIG,
        
        sample=1_000_000,
        predictions_only=pred,
        show_plots=True,
        overrides={180: "#000000"} #australia becomes black
        )

def img():
    """
    Rasterizes the entire globe using the trained NIR model.
    """
    t0 = time.perf_counter()

    raster(
        model_cfg=MODEL_CONFIG,
        checkpoint_path=MODEL_PATH,
        render = "c2",
        area="globe")
    
    dt = time.perf_counter() - t0
    print(f"Total rasterization time Elapsed: {dt:.3f}s")

if __name__ == "__main__":
    pass
    #seed_everything(SEED)
    #train()
    #viz(False)
    #img()
