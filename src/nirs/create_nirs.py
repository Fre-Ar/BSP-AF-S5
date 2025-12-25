

from .nns.nn_siren import SIRENLayer
from .nns.nn_relu import ReLULayer
from .nns.nn_incode import INCODE_NIR
from .nns.nn_gauss import GAUSSLayer
from .nns.nn_hosc import HOSCLayer
from .nns.nn_sinc import SINCLayer
from .nns.fourier_features import BasicEncoding, PositionalEncoding, RandomGaussianEncoding
from .nns.nir import MultiHeadNIR, ClassHeadConfig
from .nns.nn_split import SplitNIR
from .inference import InferenceConfig
 
from utils.utils_geo import ECOC_BITS, NUM_COUNTRIES
from utils.utils import human_int, pretty_tuple, trimf
 
def build_relu(layer_counts, class_cfg):
    model = MultiHeadNIR(
        ReLULayer,
        in_dim=3,
        layer_counts=layer_counts,
        params=None,
        class_cfg = class_cfg)
    return model

def build_siren(layer_counts, class_cfg, w0_first=30.0, w0_hidden=1.0):
    model = MultiHeadNIR(
        SIRENLayer,
        in_dim=3,
        layer_counts=layer_counts,
        params=((w0_first,),)+((w0_hidden,),)*(len(layer_counts)-1),
        class_cfg = class_cfg)
    return model

def build_incode(layer_counts, class_cfg, w0=30.0, w_hidden=1.0, learn_global_z=False):
    model = INCODE_NIR(
        in_dim=3,
        w0_first=w0,
        w0_hidden=w_hidden,
        layer_counts=layer_counts,
        class_cfg = class_cfg,
        learn_global_z = learn_global_z)
    return model

def build_gauss(layer_counts, class_cfg, s=1.0):
    model = MultiHeadNIR(
        GAUSSLayer,
        in_dim=3,
        layer_counts=layer_counts,
        params=((s,),)*(len(layer_counts)),
        class_cfg = class_cfg)
    return model

def build_hosc(layer_counts, class_cfg, beta=1.0):
    model = MultiHeadNIR(
        HOSCLayer,
        in_dim=3,
        layer_counts=layer_counts,
        params=((beta,),)*(len(layer_counts)),
        class_cfg = class_cfg)
    return model

def build_sinc(layer_counts, class_cfg, w0):
    model = MultiHeadNIR(
        SINCLayer,
        in_dim=3,
        layer_counts=layer_counts,
        params=((w0,),)*(len(layer_counts)),
        class_cfg = class_cfg)
    return model

def build_encod_basic(layer_counts, class_cfg, encoder_params):
    model = MultiHeadNIR(
        ReLULayer,
        encoder=BasicEncoding,
        encoder_params=encoder_params,
        in_dim=3,
        layer_counts=layer_counts,
        params=None,
        class_cfg = class_cfg)
    return model

def build_encod_pos(layer_counts, class_cfg, encoder_params):
    model = MultiHeadNIR(
        ReLULayer,
        encoder=PositionalEncoding,
        encoder_params=encoder_params,
        in_dim=3,
        layer_counts=layer_counts,
        params=None,
        class_cfg = class_cfg)
    return model

def build_encod_rff(layer_counts, class_cfg, encoder_params):
    model = MultiHeadNIR(
        ReLULayer,
        encoder=RandomGaussianEncoding,
        encoder_params=encoder_params,
        in_dim=3,
        layer_counts=layer_counts,
        params=None,
        class_cfg = class_cfg)
    return model

def build_split_siren(layer_counts, class_cfg, w0_first=30.0, w0_hidden=1.0):
    model = SplitNIR(
        SIRENLayer,
        in_dim=3,
        layer_counts=layer_counts,
        params=((w0_first,),)+((w0_hidden,),)*(len(layer_counts)-1),
        class_cfg = class_cfg)
    return model

def get_model_path(
    model_cfg: InferenceConfig,
    n_training=1_000_000):
    """
    Generates the standardized checkpoint path for a given model configuration.
    """
    # Base naming scheme
    layer_str = pretty_tuple(model_cfg.layer_counts)[1:-1].replace(' ', '').replace(',', '+')
    base_name = f"{model_cfg.model_name}_init-{model_cfg.init_regime}{f'_encod-{model_cfg.encoding}' if model_cfg.encoding else ''}_{model_cfg.label_mode}_{human_int(n_training)}_{layer_str}"
    suffix = ""

    match model_cfg.model_name.lower():
        case "siren" | "split_siren":
            suffix = f"_w0{trimf(model_cfg.w0)}_wh{trimf(model_cfg.w_hidden)}.pt"
        case "relu":
            suffix = ".pt"
        case "incode":
            reg = "reg_" if model_cfg.regularize_hyperparams else ""
            tiling = "global_z" if model_cfg.global_z else "RFF"
            suffix = f"_w0{trimf(model_cfg.w0)}_wh{trimf(model_cfg.w_hidden)}_{reg}{tiling}.pt"
        case "gauss":
            suffix = f"_s{trimf(model_cfg.s)}.pt"
        case "hosc":
            suffix = f"_beta{trimf(model_cfg.beta)}.pt"
        case "sinc":
            suffix = f"_w0{trimf(model_cfg.w0)}.pt"
        case 'encod_basic' | 'encod_pos' | 'encod_rff':
            suffix = f"_p{model_cfg.encoder_params}.pt"
        case _:
            raise ValueError(f"Unknown model name for path generation: {model_cfg.model_name}")
            
    return base_name + suffix

# TODO: reintroduce head layers
def build_model(
    model_cfg: InferenceConfig,
    n_training=1_000_000):
    """
    Dynamically builds a NIR.
    
    Parameters
    ----------
    model_cfg : InferenceConfig
        The configuration of the model

    Returns
    -------
    model
    """
    # 1. Generate the path using the helper
    save_path = get_model_path(model_cfg, n_training)
    
    # TODO: Make encoding play nice with everyone else

    # 2. Configure Class Heads
    cfg = ClassHeadConfig(class_mode=model_cfg.label_mode,
                        n_bits=ECOC_BITS,
                        n_classes_c1=NUM_COUNTRIES,
                        n_classes_c2=NUM_COUNTRIES)
    
    # 3. Build the specific architecture
    match model_cfg.model_name.lower():
        case "siren":
            model = build_siren(model_cfg.layer_counts, cfg, model_cfg.w0, model_cfg.w_hidden)
        case "relu":
            model = build_relu(model_cfg.layer_counts, cfg)
        case "incode":
            # params[4] is learn_global_z
            model = build_incode(model_cfg.layer_counts, cfg, model_cfg.w0, model_cfg.w_hidden, model_cfg.global_z)
        case "gauss":
            model = build_gauss(model_cfg.layer_counts, cfg, model_cfg.s)
        case "hosc":
            model = build_hosc(model_cfg.layer_counts, cfg, model_cfg.beta)
        case "sinc":
            model = build_sinc(model_cfg.layer_counts, cfg, model_cfg.w0)
        case 'encod_basic':
            model = build_encod_basic(model_cfg.layer_counts, cfg, model_cfg.encoder_params)
        case 'encod_pos':
            model = build_encod_pos(model_cfg.layer_counts, cfg, model_cfg.encoder_params)
        case 'encod_rff':
            model = build_encod_rff(model_cfg.layer_counts, cfg, model_cfg.encoder_params)
        case 'split_siren':
            model = build_split_siren(model_cfg.layer_counts, cfg, model_cfg.w0, model_cfg.w_hidden)
        case _:
            raise ValueError(f"Unknown model name: {model_cfg.model_name}")

    return model, save_path
