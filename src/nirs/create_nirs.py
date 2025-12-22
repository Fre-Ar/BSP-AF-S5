

from .nns.nn_siren import SIRENLayer
from .nns.nn_relu import ReLULayer
from .nns.nn_incode import INCODE_NIR
from .nns.nn_gauss import GAUSSLayer
from .nns.nn_hosc import HOSCLayer
from .nns.nn_sinc import SINCLayer
from .nns.fourier_features import BasicEncoding, PositionalEncoding, RandomGaussianEncoding
from .nns.nir import MultiHeadNIR, ClassHeadConfig
from .nns.nn_split import SplitNIR
 
from utils.utils_geo import ECOC_BITS, NUM_COUNTRIES
from utils.utils import human_int
 
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

def get_model_path(model_name="relu", layer_counts=(256,)*5, mode="ecoc", params=None, encoder_params=None, n_training=1_000_000, regularize_hyperparams=False):
    """
    Generates the standardized checkpoint path for a given model configuration.
    Useful for checking if a model already exists without building it.
    """
    # Base naming scheme
    base_name = f"{model_name}_{mode}_{human_int(n_training)}_{len(layer_counts)}x{layer_counts[0]}"
    suffix = ""

    match model_name.lower():
        case "siren" | "split_siren":
            suffix = f"_w0{params[0]}_wh{params[1]}.pt"
        case "relu":
            suffix = ".pt"
        case "incode":
            reg = "reg_" if regularize_hyperparams else ""
            tiling = "global_z" if params[4] else "RFF"
            suffix = f"_w0{params[0]}_wh{params[1]}_{reg}{tiling}.pt"
        case "gauss":
            suffix = f"_s{params[2]}.pt"
        case "hosc":
            suffix = f"_beta{params[3]}.pt"
        case "sinc":
            suffix = f"_w0{params[0]}.pt"
        case 'encod_basic' | 'encod_pos' | 'encod_rff':
            suffix = f"_p{encoder_params}.pt"
        case _:
            raise ValueError(f"Unknown model name for path generation: {model_name}")
            
    return base_name + suffix

# TODO: reintroduce head layers
def build_model(model_name="relu", layer_counts=(256,)*5, mode="ecoc", params=None, encoder_params=None, n_training=1_000_000, regularize_hyperparams=False):
    """
    Dynamically builds a NIR.
    
    Parameters
    ----------
    model_name : str
        Name of the architecture to use.
    layer_counts : tuple
        A tuple with the widths of the hidden layers. 
        If given (256,)*5, then the NIR will have 5 hidden layers (and thus 6 weight tensors and 6 bias tensors)
    mode : str
        Classification mode. One of 'ecoc' and 'softmax'.
    params : tuple
        Parameters for the NIRs. In the order of (w0, w_hidden, s, beta, learn_global_z)
    encoder_params : tuple
        Parameters for the encoding, in the order of (m, sigma, alpha)

    Returns
    -------
    model
    """
    # 1. Generate the path using the helper
    save_path = get_model_path(
        model_name, layer_counts, mode, params, encoder_params, n_training, regularize_hyperparams
    )

    # 2. Configure Class Heads
    cfg = ClassHeadConfig(class_mode=mode,
                        n_bits=ECOC_BITS,
                        n_classes_c1=NUM_COUNTRIES,
                        n_classes_c2=NUM_COUNTRIES)
    
    # 3. Build the specific architecture
    match model_name.lower():
        case "siren":
            model = build_siren(layer_counts, cfg, params[0], params[1])
        case "relu":
            model = build_relu(layer_counts, cfg)
        case "incode":
            # params[4] is learn_global_z
            model = build_incode(layer_counts, cfg, params[0], params[1], params[4])
        case "gauss":
            model = build_gauss(layer_counts, cfg, params[2])
        case "hosc":
            model = build_hosc(layer_counts, cfg, params[3])
        case "sinc":
            model = build_sinc(layer_counts, cfg, params[0])
        case 'encod_basic':
            model = build_encod_basic(layer_counts, cfg, encoder_params)
        case 'encod_pos':
            model = build_encod_pos(layer_counts, cfg, encoder_params)
        case 'encod_rff':
            model = build_encod_rff(layer_counts, cfg, encoder_params)
        case 'split_siren':
            model = build_split_siren(layer_counts, cfg, params[0], params[1])
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    return model, save_path
