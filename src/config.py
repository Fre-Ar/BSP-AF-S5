import math
from nirs.create_nirs import get_model_path
from nirs.inference import InferenceConfig
from utils.utils_geo import COUNTRIES_ECOC_PATH, CHECKPOINT_PATH

MODEL = "siren"
INIT_REGIME = "siren"
ENCODING = None
MODE = "softmax" 
TOTAL_LAYERS = 3 # number of total layers = depth = n_hidden + 2
WIDTH = 512
# layer_counts is the layout of the NIR trunk. 
# len(layer_counts) is the number of activation function modules.
LAYER_COUNTS = (WIDTH,)*(TOTAL_LAYERS-1)

W0 = 76.632 
WH = 4.079
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