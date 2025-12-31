# src/sweep.py
import optuna
import os
from nirs.inference import InferenceConfig
from nirs.training import train_and_eval
from utils.utils_geo import COUNTRIES_ECOC_PATH, TRAINING_DATA_PATH
from config import *
from nirs.create_nirs import get_model_size

EPOCHS = 20 # Keep low for sweeping, or use pruning
TRAINING_POINTS = 1_000_000
SIZE = "1M"
TRAIN_PATH = os.path.join(TRAINING_DATA_PATH, f"eval_uniform_{SIZE}.parquet")
EVAL_PATH = os.path.join(TRAINING_DATA_PATH, f"eval_uniform_1M.parquet")

def objective(trial):
    
    # 1. Architecture Sweep
    width = trial.suggest_categorical("width", [128, 256, 512, 1024])
    
    # Dynamically calculate the Max Depth allowed for this Width
    valid_depths = []
    possible_depths = range(3, 9) # 3 to 8
    for d in possible_depths:
        params = get_model_size(d, width)
        if params <= 2_000_000: # 8MB limit (float32)
            valid_depths.append(d)
        else:
            # Since params increase with depth, we can stop early
            break
            
    if not valid_depths:
    #    # This acts as a safety valve if even Depth=3 is too big
        raise optuna.TrialPruned()
    
    # Suggest Depth from the VALID range only
    max_safe_depth = max(valid_depths)
    depth = trial.suggest_int("depth", 3, max_safe_depth)
    
    #depth = trial.suggest_int("depth", 3, 15)
    layer_counts = (width,) * (depth - 1)
    
    # 2. SIREN Hyperparams Sweep
    # w0: Frequency multiplier for FIRST layer (Standard: ~30.0)
    w0 = trial.suggest_float("w0", 20.0, 150.0)
    
    # w_hidden: Frequency multiplier for HIDDEN layers (Standard: 1.0)
    # Raising this increases the "high frequency" capacity of deep layers.
    w_hidden = trial.suggest_float("w_hidden", 0.8, 15.0)
    
    # Optimization
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True) 
    
    # Weight Decay
    # Very sensitive parameter. We sweep log-scale from near-zero to strong.
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    
    # 3. Construct Config
    model_cfg = InferenceConfig(
        model_name="siren",
        init_regime="siren",
        encoding=None,
        layer_counts=layer_counts,
        
        w0=w0, 
        w_hidden=w_hidden,
        s=S, 
        beta=BETA, 
        k=K,
        
        global_z=GLOBAL_Z, 
        regularize_hyperparams=REG_HYPER,
        
        # Unused params for SIREN but required by config
        FR_f=FR_F, FR_p=FR_P, FR_alpha=FR_ALPHA,
        encod_alpha=ENCOD_ALPHA, encod_sigma=ENCOD_SIGMA, encod_m=ENCOD_M,
        
        label_mode="softmax", 
        codes_path=COUNTRIES_ECOC_PATH
    )
    
    # 4. Run Training
    try:
        score = train_and_eval(
            train_set_path=TRAIN_PATH,
            #eval_set_path=EVAL_PATH,
            model_cfg=model_cfg,
            epochs=EPOCHS,
            traning_size=TRAINING_POINTS,
            batch_size=8192,
            lr=lr,
            weight_decay=weight_decay,
            use_uncertainty_loss_weighting=True, 
            loss_weights=(1.0, 1.0, 1.0),
            device="mps", # or "cuda"
            trial=trial   # Pass trial for pruning
        )
    except RuntimeError as e:
        # Catch OOM errors so one bad config doesn't kill the sweep
        if "out of memory" in str(e).lower():
            print(f"Trial failed: OOM")
            return float('inf')
        raise e

    return score

if __name__ == "__main__":
    # 5. Setup Study
    storage_url = "sqlite:///db.sqlite3" # Saves progress to file
    study = optuna.create_study(
        study_name="siren_sweep_1M_uw",
        direction="minimize",
        storage=storage_url,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=8)
    )
    
    print(f"Running sweep... View results with: optuna-dashboard {storage_url}")
    study.optimize(objective, n_trials=150)
    
    print("Best params:", study.best_params)