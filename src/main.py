# src/main.py
import os
import time

import torch

from nirs.viz.compare_data_viz import visualize_model
from nirs.viz.rasterizer import raster
from nirs.training import train_and_eval
from utils.utils_geo import TRAINING_DATA_PATH, BEST_LOGS_PATH
from config import *

def train():
    """
    On MPS:
        1mb model:
            20 epochs -> 150s (99s if plugged in)
        4mb model:
            20 epochs -> 240s
    """

    t0 = time.perf_counter()
    train_and_eval(
        train_dir=TRAIN_DIR,
        model_cfg=MODEL_CONFIG,
        eval_set_path=EVAL_PATH,
        #out_dir=BEST_CHECKPOINT_PATH,
        #log_dir=BEST_LOGS_PATH,
        batch_size = 16384,
        traning_size = TRAINING_POINTS,
        lr=LR,
        weight_decay=WD,
        #lr=3e-4,
        )
    dt = time.perf_counter() - t0
    print(f"Total training time Elapsed: {dt:.3f}s")

def viz(pred: bool = False):  
    visualize_model(
        #parquet_path=os.path.join(TRAINING_DATA_PATH, "log_dataset_1M.parquet"),
        parquet_path=os.path.join(TRAINING_DATA_PATH, "eval_uniform_1M.parquet"),
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
        area="uk")
    
    dt = time.perf_counter() - t0
    print(f"Total rasterization time Elapsed: {dt:.3f}s")

def get_counts():
    from nirs.weights import compute_class_counts
    
    def tensor_to_dict(tensor: torch.Tensor) -> dict:
        """
        Converts a tensor to a dictionary where keys are '1-based index' strings.
        Example: tensor([0.5, 0.9]) -> {'1': 0.5, '2': 0.9}
        """
        return {str(i + 1): value.item() for i, value in enumerate(tensor)}
    
    c1_counts = compute_class_counts(
        data_dir=TRAIN_DIR,
        id_col="c1_id"
    )
    c2_counts = compute_class_counts(
        data_dir=TRAIN_DIR,
        id_col="c2_id"
    )
    print("C1 counts:", tensor_to_dict(c1_counts))
    print("-------------------")
    print("C2 counts:", tensor_to_dict(c2_counts))
    
    

if __name__ == "__main__":
    #pass
    #train()
    #viz(True)
    img()
