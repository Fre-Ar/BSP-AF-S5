# src/main.py
import os
import time

from nirs.viz.compare_data_viz import compare_parquet_and_model_ecoc
from nirs.viz.rasterizer import raster
from nirs.training import train_and_eval
from nirs.create_nirs import get_model_path

from utils.utils_geo import COUNTRIES_ECOC_PATH, TRAINING_DATA_PATH, CHECKPOINT_PATH

MODEL = "incode"
INIT_REGIME = "siren"
ENCODING = None
MODE = "ecoc" 
DEPTH = 5
LAYER = 512
LAYER_COUNTS = (LAYER,)*DEPTH

W0 = 30.0 
WH = 1.0
S = 1.0
BETA = 1.0
GLOBAL_Z = False # False enables RFF latent Z code
REG_HYPER = True

TRAINING_POINTS = 1_000_000

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
        epochs=20,
        batch_size = 8192,
        #lr=3e-4,
        
        model_name=MODEL,
        init_regime=INIT_REGIME,
        encoding=ENCODING,
        layer_counts=LAYER_COUNTS,
        
        label_mode=MODE,

        w0=W0,
        w_hidden=WH,
        beta=BETA,
        s=S,
        
        regularize_hyperparams=REG_HYPER,
        global_z=GLOBAL_Z
        )
    dt = time.perf_counter() - t0
    print(f"Total training time Elapsed: {dt:.3f}s")

def viz(pred: bool = False):
    model_path = get_model_path(
        model_name=MODEL,
        layer_counts=LAYER_COUNTS,
        mode=MODE,
        params=(W0, WH, S, BETA, GLOBAL_Z),
        encoder_params=None,
        n_training=TRAINING_POINTS,
        regularize_hyperparams=REG_HYPER)    
    model_path = f"{CHECKPOINT_PATH}/{model_path}" 
    
    compare_parquet_and_model_ecoc(
        parquet_path=os.path.join(TRAINING_DATA_PATH, "log_dataset_1M.parquet"),
        #parquet_path=os.path.join(TRAINING_DATA_PATH, "eval_uniform_1M.parquet"),
        #parquet_path=os.path.join(TRAINING_DATA_PATH, "eval_border_1M.parquet"),
    
        checkpoint_path=model_path,
        model_name=MODEL,
        
        label_mode=MODE,
        codes_path=COUNTRIES_ECOC_PATH,
        
        sample=1_000_000,
        model_outputs_log1p=True,
        
        predictions_only=pred,
        
        overrides={180: "#000000"}, #australia becomes black
        
        layer_counts=LAYER_COUNTS,
        depth=DEPTH,
        layer=LAYER,
        w0=W0, w_h=WH, s_param=S, beta=BETA, global_z=GLOBAL_Z,
        
        regularize_hyperparams=REG_HYPER)

def img():
    t0 = time.perf_counter()

    raster(MODEL,
        MODE,
        LAYER_COUNTS,
        DEPTH,
        LAYER,
        W0, WH, S, BETA, GLOBAL_Z,
        REG_HYPER,
        render = "c1",
        area="alpes")
    
    dt = time.perf_counter() - t0
    print(f"Total rasterization time Elapsed: {dt:.3f}s")

if __name__ == "__main__":
    train()
    #viz()
    #img()