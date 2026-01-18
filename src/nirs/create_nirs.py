
import math
from typing import Tuple, Type, Optional, Callable, Dict

import torch.nn as nn
from .nns.nn_siren import SIRENLayer, Sine
from .nns.nn_relu import ReLULayer
from .nns.nn_incode import INCODE_NIR
from .nns.nn_gauss import GAUSSLayer
from .nns.nn_hosc import HOSCLayer
from .nns.nn_sinc import SINCLayer
from .nns.nn_finer import FINERLayer
from .nns.nn_wire import WIRELayer
from .nns.nn_mfn import MFN_NIR
from .nns.nn_fr import FR_NIR
from .nns.fourier_features import BasicEncoding, PositionalEncoding, RandomGaussianEncoding
from .nns.nir import MultiHeadNIR, ClassHeadConfig
from .nns.nn_split import SplitNIR
from .inference import InferenceConfig
from .nns.init_regimes import (
    init_siren_linear,
    init_linear,
    init_reset,
    init_none,
    init_finer_linear,
    init_mfn_linear,
    init_mfn_filter)
 
from utils.utils_geo import ECOC_BITS, NUM_COUNTRIES
from utils.utils import human_int, pretty_tuple, to_scientific, trimf

# -----------------------------------------------------------------------------
# REGISTRIES
# -----------------------------------------------------------------------------

# Maps config string names to Layer Classes
LAYER_REGISTRY: Dict[str, Type] = {
    "siren": SIRENLayer,
    "relu": ReLULayer,
    "gauss": GAUSSLayer,
    "hosc": HOSCLayer,
    "sinc": SINCLayer,
    "finer": FINERLayer,
    "wire": WIRELayer,
}

# Maps config string names to Activation Function Classes
ACT_REGISTRY: Dict[str, Type] = {
    "siren": Sine,
    "relu": nn.ReLU,
}

# Maps config string names to Encoder Classes
ENCODER_REGISTRY: Dict[str, Type] = {
    "basic": BasicEncoding,
    "pos": PositionalEncoding,
    "rff": RandomGaussianEncoding,
}

# Maps config string names to Init linear functions
INIT_REGISTRY: Dict[str, Callable] = {
    "siren": init_siren_linear,
    "default": init_linear,
    "reset": init_reset,
    "none": init_none,
    "finer": init_finer_linear,
    "mfn": init_mfn_linear,
}

# -----------------------------------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------------------------------

def _get_layer_params(cfg: InferenceConfig) -> Optional[Tuple[Tuple[float, ...], ...]]:
    """
    Constructs the layer-wise parameter tuples required by MultiHeadNIR.
    Handles the logic for First Layer vs Hidden Layers (e.g. SIREN w0).
    """
    n_layers = len(cfg.layer_counts) # trunk layers
    
    match cfg.model_name.lower():
        case "siren" | "split_siren":
            # SIREN: (w0_first) for layer 0, (w_hidden) for rest
            return ((cfg.w0,),) + ((cfg.w_hidden,),) * (n_layers - 1)
        
        case "gauss":
            # GAUSS: (s) for all layers
            return ((cfg.s,),) * n_layers
            
        case "hosc":
            # HOSC:  (w0_first, beta) for layer 0, (w_hidden, beta) for rest
            return ((cfg.w0, cfg.beta),) + ((cfg.w_hidden, cfg.beta),) * (n_layers - 1)

        case "sinc":
            # SINC: (w0) for all layers 
            return ((cfg.w0,),) * n_layers
        
        case "wire":
            # WIRE: (w,s) for all layers
            return ((cfg.w0, cfg.s),) * n_layers
        
        case "finer":
            # FINER: (w,k) for all layers
            return ((cfg.w0,cfg.k),) * n_layers
            
        case "relu":
            # ReLU doesn't need per-layer params
            return None
            
        case _:
            raise ValueError(f"No parameter extraction logic defined for {cfg.model_name}")

# -----------------------------------------------------------------------------
# SPECIFIC BUILDERS
# -----------------------------------------------------------------------------

def _build_standard_nir(cfg: InferenceConfig, class_head_cfg: ClassHeadConfig):
    """Builds standard MLP-based NIRs (SIREN, ReLU, Gauss, etc)."""
    nir_layer = LAYER_REGISTRY.get(cfg.model_name.lower())
    if not nir_layer:
        raise ValueError(f"Unknown standard layer type: {cfg.model_name}")
    
    init_regime = INIT_REGISTRY.get(cfg.init_regime.lower()) if cfg.init_regime else None
    
    encoder = ENCODER_REGISTRY.get(cfg.encoding.lower()) if cfg.encoding else None
    
    return MultiHeadNIR(
        layer=nir_layer,
        init_regime=init_regime,
        encoder=encoder,
        in_dim=3,
        
        layer_counts=cfg.layer_counts,
        params=_get_layer_params(cfg),
        
        encoder_params=(cfg.encod_alpha, cfg.encod_sigma, cfg.encod_m),
        class_cfg=class_head_cfg
    )

def _build_incode(cfg: InferenceConfig, class_head_cfg: ClassHeadConfig):
    """Builds the INCODE architecture."""
    init_regime = INIT_REGISTRY.get(cfg.init_regime.lower()) if cfg.init_regime else None
    
    return INCODE_NIR(
        init_regime=init_regime,
        in_dim=3,
        w0_first=cfg.w0,
        w0_hidden=cfg.w_hidden,
        layer_counts=cfg.layer_counts,
        class_cfg=class_head_cfg,
        learn_global_z=cfg.global_z
    )

def _build_split_nir(cfg: InferenceConfig, class_head_cfg: ClassHeadConfig):
    """Builds the Split architecture (SIREN based)."""
    init_regime = INIT_REGISTRY.get(cfg.init_regime.lower()) if cfg.init_regime else None
    return SplitNIR(
        layer=SIRENLayer,
        init_regime=init_regime,
        in_dim=3,
        layer_counts=cfg.layer_counts,
        params=_get_layer_params(cfg),
        class_cfg=class_head_cfg
    )
    
def _build_mfn(cfg: InferenceConfig, class_head_cfg: ClassHeadConfig):
    """Builds the MFN architecture."""
    filter_name = cfg.model_name.split('_')[1]
    init_regime = INIT_REGISTRY.get(cfg.init_regime.lower()) if cfg.init_regime else None

    return MFN_NIR(
        in_dim=3,
        width=cfg.layer_counts[0],
        depth=len(cfg.layer_counts)+1,
        filter_type=filter_name,
        weight_scale=1.0,
        linear_init_regime=init_regime,
        filter_init_regime=init_mfn_filter,
        class_cfg=class_head_cfg
    )
    
def _build_fr(cfg: InferenceConfig, class_head_cfg: ClassHeadConfig):
    """Builds the FR architecture."""
    act_name = cfg.model_name.split('_')[1]
    act = ACT_REGISTRY.get(act_name)
    init_regime = INIT_REGISTRY.get(cfg.init_regime.lower()) if cfg.init_regime else None

    act_params = (cfg.w0,cfg.w_hidden) if (act_name == "siren") else ()
    
    return FR_NIR(
        activation=act,
        init_regime=init_regime,
        in_dim=3,
        hidden_dim=cfg.layer_counts[0],
        depth=len(cfg.layer_counts)+1,
        freq=cfg.FR_f,
        phases=cfg.FR_p,
        alpha=cfg.FR_alpha,
        params=act_params,
        class_cfg=class_head_cfg
    )

def get_model_path(
    model_cfg: InferenceConfig,
    n_training=1_000_000,
    lr = 1e-4):
    """
    Generates the standardized checkpoint path for a given model configuration.
    """
    # Base naming scheme
    layer_str = pretty_tuple(model_cfg.layer_counts)[1:-1].replace(' ', '').replace(',', '+')
    enc_str = f"_encod-{model_cfg.encoding}" if model_cfg.encoding else ""
    
    base_name = (
        f"{model_cfg.model_name}_"
        f"lr={to_scientific(lr)}_"
        f"init-{model_cfg.init_regime}"
        f"{enc_str}_"
        f"{model_cfg.label_mode}_"
        f"{human_int(n_training)}_"
        f"{layer_str}"
    )
    suffix = ""
    
    match model_cfg.model_name.lower():
        case "siren" | "split_siren":
            suffix = f"_w0={trimf(model_cfg.w0)}_wh={trimf(model_cfg.w_hidden)}"
        case "incode":
            reg = "reg_" if model_cfg.regularize_hyperparams else ""
            tiling = "global_z" if model_cfg.global_z else "RFF"
            suffix = f"_w0={trimf(model_cfg.w0)}_wh={trimf(model_cfg.w_hidden)}_{reg}{tiling}"
        case "gauss":
            suffix = f"_s={trimf(model_cfg.s)}"
        case "hosc":
            suffix = f"_beta={trimf(model_cfg.beta)}"
        case "sinc":
            suffix = f"_w0={trimf(model_cfg.w0)}"
        case "relu":
            suffix = ""
        case "fr_siren" | "fr_relu":
            suffix = (
                f"_w0={trimf(model_cfg.w0)}_"
                f"freq={human_int(model_cfg.FR_f)}_"
                f"phases={human_int(model_cfg.FR_p)}_"
                f"fr-alpha={trimf(model_cfg.FR_alpha)}"
            )
        case _:
            raise ValueError(f"Unknown model name for path generation: {model_cfg.model_name}")
    
    if model_cfg.encoding and model_cfg.encoding in ENCODER_REGISTRY:
        enc_params = (
            f"_m={trimf(model_cfg.encod_m)}"
            f"_sigma={trimf(model_cfg.encod_sigma)}"
            f"_alpha={trimf(model_cfg.encod_alpha)}"
        )
        suffix += f"{enc_params}"
        
    return f"{base_name}{suffix}.pt"

def build_model(
    model_cfg: InferenceConfig,
    n_training=1_000_000,
    lr = 1e-4):
    """
    Dynamically builds a NIR.
    
    Parameters
    ----------
    model_cfg : InferenceConfig
        The configuration of the model

    Returns
    -------
        (model, save_path_string)
    """
    # 1. Generate the path using the helper
    save_path = get_model_path(model_cfg, n_training, lr)
    

    # 2. Configure Class Heads
    class_cfg = ClassHeadConfig(class_mode=model_cfg.label_mode,
                        n_bits=ECOC_BITS,
                        n_classes_c1=NUM_COUNTRIES,
                        n_classes_c2=NUM_COUNTRIES)
    
    # 3. Dispatch to specific builder
    name = model_cfg.model_name.lower()

    if name == "incode":
        model = _build_incode(model_cfg, class_cfg)
    elif name == "split_siren":
        model = _build_split_nir(model_cfg, class_cfg)
    elif name in LAYER_REGISTRY:
        model = _build_standard_nir(model_cfg, class_cfg)
    elif name.startswith('mfn'): # mfn_fourier or mfn_gabor
        model = _build_mfn(model_cfg, class_cfg)
    elif name.startswith('fr_'): # fr_relu or fr_siren
        model = _build_fr(model_cfg, class_cfg)
    else:
        raise ValueError(f"Unknown model name: {model_cfg.model_name}")
    
    return model, save_path


def get_model_size(depth: int, width: int, flag: str = "") -> int:
    if flag == "split":
        return 12 * width + 3*(depth-2)*(width * width + width) + (width+1) * (NUM_COUNTRIES*2 + 1)
    else:
        return 4 * width + (depth-2)*(width * width + width) + (width+1) * (NUM_COUNTRIES*2 + 1)