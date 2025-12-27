# src/main.py
import os
import time
import math

from nirs.viz.compare_data_viz import visualize_model
from nirs.viz.rasterizer import raster
from nirs.training import train_and_eval
from nirs.create_nirs import get_model_path
from nirs.inference import InferenceConfig
from utils.utils import get_default_device

from utils.utils_geo import COUNTRIES_ECOC_PATH, TRAINING_DATA_PATH, CHECKPOINT_PATH

MODEL = "siren"
INIT_REGIME = "siren"
ENCODING = None
MODE = "softmax" 
TOTAL_LAYERS = 5 # number of total layers = depth = n_hidden + 2
WIDTH = 256
# layer_counts is the layout of the NIR trunk. 
# len(layer_counts) is the number of activation function modules.
LAYER_COUNTS = (WIDTH,)*(TOTAL_LAYERS-1)

W0 = 70.0 
WH = 1.0
S = 7.07
BETA = 8.0
K = 20.0
GLOBAL_Z = False # False enables RFF latent Z code
REG_HYPER = True
FR_F = 256
FR_P = 8
FR_ALPHA = 0.01

ENCOD_ALPHA = 2.0 * math.pi
ENCOD_SIGMA = 5.0
ENCOD_M = 256

MODEL_CONFIG = InferenceConfig(
    MODEL, INIT_REGIME, ENCODING,
    LAYER_COUNTS, 
    W0, WH, S, BETA, K,
    GLOBAL_Z, REG_HYPER,
    FR_F ,FR_P, FR_ALPHA,
    ENCOD_ALPHA, ENCOD_SIGMA, ENCOD_M,
    MODE, COUNTRIES_ECOC_PATH
)

TRAINING_POINTS = 1_000_000
EPOCHS = 20

model_path = get_model_path(
        model_cfg=MODEL_CONFIG,
        n_training=TRAINING_POINTS,
        max_epochs=EPOCHS)    
MODEL_PATH = f"{CHECKPOINT_PATH}/{model_path}" 

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
    PATH = os.path.join(TRAINING_DATA_PATH, f"eval_uniform_{SIZE}.parquet")
    #PATH = os.path.join(TRAINING_DATA_PATH, f"log_dataset_{SIZE}.parquet")
    #PATH = os.path.join(TRAINING_DATA_PATH, f"training_{SIZE}.parquet")
    
    t0 = time.perf_counter()
    train_and_eval(
        PATH,
        model_cfg=MODEL_CONFIG,
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
    train()
    #viz()
    #img()