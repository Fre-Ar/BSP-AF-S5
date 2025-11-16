# src/main.py
import torch
from torch.utils.data import DataLoader
import math 
import pathlib
import os

from nirs.viz.compare_data_viz import compare_parquet_and_model_ecoc
from nirs.viz.rasterizer import raster
from nirs.training import train_and_eval
from nirs.world_bank_country_colors import colors_important

from utils.utils_geo import COUNTRIES_ECOC_PATH, TRAINING_DATA_PATH, CHECKPOINT_PATH

MODEL = "split_siren"
MODE = "ecoc" 
DEPTH = 15
LAYER = 128
LAYER_COUNTS = (LAYER,)*DEPTH

W0 = 30.0 
WH = 1.0
S = 1.0
BETA = 1.0
GLOBAL_Z = True
REG_HYPER = True

def train():
    PATH = os.path.join(TRAINING_DATA_PATH, "log_dataset_1M.parquet")
    
    train_and_eval(
        PATH,
        epochs=100,
        
        model_name="split_siren",
        layer_counts=LAYER_COUNTS,
        
        label_mode=MODE,

        w0=W0,
        w_hidden=WH,
        beta=BETA,
        s=S,
        
        regularize_hyperparams=REG_HYPER,
        global_z=GLOBAL_Z)

def viz():
    model_path = f"{CHECKPOINT_PATH}/{MODEL}_{MODE}_1M_{DEPTH}x{LAYER}_w0{W0}_wh{WH}.pt" 
    
    compare_parquet_and_model_ecoc(
        parquet_path=os.path.join(TRAINING_DATA_PATH, "log_dataset_1M.parquet"),
        checkpoint_path=model_path,
        model_name=MODEL,
        
        label_mode=MODE,
        codes_path=COUNTRIES_ECOC_PATH,
        
        sample=1_000_000,
        model_outputs_log1p=True,
        
        predictions_only=False,
        overrides={180: "#000000"}, #australia becomes black
        
        layer_counts=LAYER_COUNTS,
        depth=DEPTH,
        layer=LAYER,
        w0=W0, w_h=WH, s_param=S, beta=BETA, global_z=GLOBAL_Z,
        
        regularize_hyperparams=REG_HYPER)

def img():
    raster(MODEL,
        MODE,
        LAYER_COUNTS,
        DEPTH,
        LAYER,
        W0, WH, S, BETA, GLOBAL_Z,
        REG_HYPER,
        render = "c1",
        area="lux")

    
if __name__ == "__main__":
    train()
    #viz()
    #img()