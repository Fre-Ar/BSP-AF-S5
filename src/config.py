import math
import os
from nirs.create_nirs import get_model_path
from nirs.inference import InferenceConfig
from utils.utils_geo import COUNTRIES_ECOC_PATH, CHECKPOINT_PATH, TRAINING_DATA_PATH, BEST_CHECKPOINT_PATH

MODEL = "split_siren"
INIT_REGIME = "siren"
ENCODING = None
MODE = "softmax" 
TOTAL_LAYERS = 5 # number of total layers = depth = n_hidden + 2
WIDTH = 256
# layer_counts is the layout of the NIR trunk. 
# len(layer_counts) is the number of activation function modules.
LAYER_COUNTS = (WIDTH,)*(TOTAL_LAYERS-1)

#W0 = 76.632 
#WH = 4.079

#W0 = 51.959
#WH = 2.59

W0 = 30.0#56.4
WH = W0#7.1
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

TRAINING_POINTS = 200_000_000
LR = 1e-4#5.22e-5
WD = 0

model_path = get_model_path(
        model_cfg=MODEL_CONFIG,
        n_training=TRAINING_POINTS)    
MODEL_PATH = f"{BEST_CHECKPOINT_PATH}/{model_path}" 
TRAIN_DIR = os.path.join(TRAINING_DATA_PATH, "training")
TRAIN_BIAS_DIR = os.path.join(TRAINING_DATA_PATH, "training_biased")
EVAL_PATH = os.path.join(TRAINING_DATA_PATH, f"eval_uniform_1M.parquet")
