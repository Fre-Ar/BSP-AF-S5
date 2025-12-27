# src/nirs/inference.py

import numpy as np
from dataclasses import dataclass
from typing import Optional

from utils.utils_geo import COUNTRIES_ECOC_PATH
from .nns.nir import LabelMode

import math


@dataclass
class InferenceConfig:
    """
    Configuration for a NIR model.
    """
    model_name: str
    init_regime: str | None = None # None = default
    encoding: str | None = None
    layer_counts: tuple = (256,)*5 # layer_counts is the layout of the NIR trunk. 
    # n_hidden layers will always be len(layer_counts)-1
    # len(layer_counts) is the number of activation function modules.

    # Model Hyperparameters
    w0: float = 30.0
    w_hidden: float = 1.0
    s: float = 1.0
    beta: float = 1.0
    k: float =  20.0
    global_z: bool = False
    regularize_hyperparams: bool = False
    FR_f: float = 256.0
    FR_p: float = 8.0
    FR_alpha: float = 0.01
    
    encod_alpha: float = 2.0 * math.pi
    encod_sigma: float = 5.0
    encod_m: int = 256
    
    # Inference Settings
    label_mode: LabelMode = "ecoc"   # "ecoc" or "softmax"
    codes_path: Optional[str] = COUNTRIES_ECOC_PATH
    model_outputs_log1p: bool = True

@dataclass
class Prediction:
    """
    Standardized output container for the Predictor.
    All arrays are on CPU.
    """
    dist_km: np.ndarray      # Distance in km
    c1_ids: np.ndarray       # Class ID 1
    c2_ids: np.ndarray       # Class ID 2
    logits_c1: Optional[np.ndarray] = None 
    logits_c2: Optional[np.ndarray] = None


