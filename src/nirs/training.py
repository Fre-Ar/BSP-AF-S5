# src/nirs/training.py

import math
import pathlib
import time
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch import cat

from .loss import UncertaintyWeighting
from .nns.nn_incode import INCODE_NIR as Incode
from .nns.nir import LabelMode, ClassHeadConfig
from .create_nirs import build_model
from .engine import compute_potential_ecoc_pos_weights
from .data import make_dataloaders
from .metrics import compute_distance_metrics, compute_classification_metrics

from geodata.ecoc.ecoc import per_bit_threshold, _prepare_codebook_tensor
from utils.utils import get_default_device, trimf, pretty_tuple
from utils.utils_geo import COUNTRIES_ECOC_PATH, CHECKPOINT_PATH, TRAINING_LOG_PATH

# ===================== CONFIGURATION =====================

@dataclass
class LossWeights:
    w_dist: float = 1.0
    w_c1: float = 1.0
    w_c2: float = 1.0

# ===================== TRAINER CLASS =====================

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str,
        loss_weights: LossWeights,
        # Data Config
        class_cfg: ClassHeadConfig,
        pos_weight_c1: Optional[Tensor] = None,
        pos_weight_c2: Optional[Tensor] = None,
        codebook: Optional[Dict[int, np.ndarray]] = None,
        # Advanced
        uw: Optional[UncertaintyWeighting] = None,
        regularize_hyperparams: bool = False
    ):
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.device = device
        self.lw = loss_weights
        
        self.class_cfg = class_cfg
        self.pos_weight_c1 = pos_weight_c1.to(device) if pos_weight_c1 is not None else None
        self.pos_weight_c2 = pos_weight_c2.to(device) if pos_weight_c2 is not None else None
        
        
        class_ids, codes_mat = _prepare_codebook_tensor(codebook, device) if codebook else (None, None)
        self.codes_mat = codes_mat
        self.class_ids = class_ids
        
        self.uw = uw
        self.regularize_hyperparams = regularize_hyperparams

        # Pre-instantiate loss functions to avoid re-creation
        self.mse_loss = nn.MSELoss(reduction="mean").to(device)
        self.ce_loss = nn.CrossEntropyLoss(reduction="mean").to(device)
        
        # BCE with weights (if provided)
        self.bce_c1 = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight_c1).to(device)
        self.bce_c2 = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight_c2).to(device)

        # Cache thresholds for ECOC evaluation to save compute
        self.thr1 = None
        self.thr2 = None

    def _get_thresholds(self, n_bits_c1: int, n_bits_c2: int):
        """Lazy load thresholds once."""
        if self.thr1 is None:
            self.thr1 = per_bit_threshold(self.pos_weight_c1, self.device, n_bits_c1)
        if self.thr2 is None:
            self.thr2 = per_bit_threshold(self.pos_weight_c2, self.device, n_bits_c2)
        return self.thr1, self.thr2

    def _forward(self, xyz: Tensor):
        """
        Unified forward pass for all NIR models.

        For INCODE, returns:
        (pred_log1p_dist, c1_logits, c2_logits, (a, b, c, d))

        For all other models, returns:
        (pred_log1p_dist, c1_logits, c2_logits, None)
        """
        # INCODE has a slightly different signature and returns extra hyperparams
        if self.model_name.lower() == "incode":
            output = self.model(xyz, self.regularize_hyperparams)
            if self.regularize_hyperparams:
                # unpack: (dist, c1, c2, (a,b,c,d))
                return output[0], output[1], output[2], output[3]
            else:
                # unpack: (dist, c1, c2)
                return output[0], output[1], output[2], None
        else:
            # Standard models
            pred_dist, c1, c2 = self.model(xyz)
            return pred_dist, c1, c2, None
        
    def _compute_losses(self, pred_dist, c1_logits, c2_logits, batch):
        """
        Computes losses for all 3 heads, handling ECOC vs softmax label mode.

        Returns
        -------
        loss_dist, loss_c1, loss_c2 : scalar tensors
        """
        # Distance Loss (Log space)
        gt_log1p = batch["log1p_dist"].to(self.device, non_blocking=True)
        loss_dist = self.mse_loss(pred_dist, gt_log1p)

        # Class Loss
        if self.class_cfg.class_mode == "ecoc":
            loss_c1 = self.bce_c1(c1_logits, batch["c1_bits"].to(self.device))
            loss_c2 = self.bce_c2(c2_logits, batch["c2_bits"].to(self.device))
        else:
            loss_c1 = self.ce_loss(c1_logits, batch["c1_idx"].to(self.device))
            loss_c2 = self.ce_loss(c2_logits, batch["c2_idx"].to(self.device))
            
        return loss_dist, loss_c1, loss_c2
    
    def train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        
        # For debug stats
        total_loss = 0.0
        d_loss = 0.0
        c1_loss = 0.0
        c2_loss = 0.0
        sum_grad_norm = 0.0
        n_batches = 0

        pbar = tqdm(loader, leave=False, desc="Train")
        for batch in pbar:
            # 1. Measure Data Loading
            # Time elapsed since the last loop ended is pure data waiting time
            xyz = batch["xyz"].to(self.device, non_blocking=True)
            
            # 1. Zero Grad
            self.optimizer.zero_grad(set_to_none=True)

            # 2. Forward
            pred_log1p, c1_logits, c2_logits, reg_params = self._forward(xyz)
            
            # 3. Compute Losses
            l_dist, l_c1, l_c2 = self._compute_losses(pred_log1p, c1_logits, c2_logits, batch)

            # 4. Weighting
            if self.uw:
                loss = self.uw([self.lw.w_dist * l_dist, self.lw.w_c1 * l_c1, self.lw.w_c2 * l_c2])
            else:
                loss = (self.lw.w_dist * l_dist) + (self.lw.w_c1 * l_c1) + (self.lw.w_c2 * l_c2)

            if reg_params is not None:
                # INCODE regularization
                loss += Incode.incode_reg(*reg_params)
                
            # 5. Backward
            loss.backward()

            # 6. Grad Norm
            # Compute norm over all parameters with gradients
            gn = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=math.inf) 
            sum_grad_norm += gn.item()

            self.optimizer.step()
            
            # 7. Stats 
            total_loss += loss.item()
            d_loss += l_dist.item()
            c1_loss += l_c1.item()
            c2_loss += l_c2.item()
            n_batches += 1
            pbar.set_postfix({
                "loss": f"{trimf(loss.item(),4)}",
                "d_loss": f"{trimf(l_dist.item(),4)}",
                "c1_loss": f"{trimf(l_c1.item(),4)}",
                "c2_loss": f"{trimf(l_c2.item(),4)}"
            })
            
        inv_n =  1/max(1, n_batches)
        return {
            "loss": f"{trimf(total_loss*inv_n,4)}",
            "d_loss":  f"{trimf(d_loss*inv_n,4)}",
            "c1_loss": f"{trimf(c1_loss*inv_n,4)}",
            "c2_loss": f"{trimf(c2_loss*inv_n,4)}",
            "grad_norm": sum_grad_norm * inv_n
        }

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        
        # Collectors (gt = ground truth)
        all_pred_dist = []
        all_gt_dist = [] # in km
        
        all_c1_logits = []
        all_c1_gt = []   # ids
        all_c1_bits = [] # For BitAcc if ECOC
        
        all_c2_logits = []
        all_c2_gt = []
        all_c2_bits = []
        
        for batch in loader:
            xyz = batch["xyz"].to(self.device, non_blocking=True)

            # Forward
            pred_log1p, c1_logits, c2_logits, _ = self._forward(xyz)
            
            # Move predictions to CPU immediately for storage
            # Distance
            all_pred_dist.append(torch.expm1(pred_log1p).cpu())
            all_gt_dist.append(torch.expm1(batch["log1p_dist"]).cpu())
            
            
            # Classification
            all_c1_logits.append(c1_logits.cpu())
            if "c1_idx" in batch: all_c1_gt.append(batch["c1_idx"].cpu())
            if "c1_bits" in batch: all_c1_bits.append(batch["c1_bits"].cpu())
            
            all_c2_logits.append(c2_logits.cpu())
            if "c2_idx" in batch: all_c2_gt.append(batch["c2_idx"].cpu())
            if "c2_bits" in batch: all_c2_bits.append(batch["c2_bits"].cpu())
            
        
        # Concatenate on CPU
        # Optimize
        pred_d = cat(all_pred_dist)
        gt_d = cat(all_gt_dist)
        
        c1_log = cat(all_c1_logits)
        c2_log = cat(all_c2_logits)
        c1_gt = cat(all_c1_gt) if all_c1_gt else None
        c2_gt = cat(all_c2_gt) if all_c2_gt else None
        
        # --- Compute Metrics ---
        stats = {}
        
        # 1. Distance
        d_metrics = compute_distance_metrics(pred_d.flatten(), gt_d.flatten())
        stats.update(d_metrics)
        
        # 2. Classification C1
        device_calc = self.device
        
        if c1_gt is not None:
            c1_bits_tensor = cat(all_c1_bits).to(device_calc) if all_c1_bits else None
            
            c1_met = compute_classification_metrics(
                c1_log.to(device_calc), c1_gt.to(device_calc), 
                self.class_cfg.class_mode, 
                self.codes_mat,
                self.class_ids,
                self.pos_weight_c1,
                c1_bits_tensor
            )
            # Prefix keys
            for k, v in c1_met.items(): stats[f"c1_{k}"] = v
            
        # 3. Classification C2
        if c2_gt is not None:
            c2_bits_tensor = cat(all_c2_bits).to(device_calc) if all_c2_bits else None
            
            c2_met = compute_classification_metrics(
                c2_log.to(device_calc), c2_gt.to(device_calc), 
                self.class_cfg.class_mode, 
                self.codes_mat,
                self.class_ids,
                self.pos_weight_c2,
                c2_bits_tensor
            )
            for k, v in c2_met.items(): stats[f"c2_{k}"] = v
        
        return stats
    
def setup_logging(
    out_dir: str | os.PathLike,
    log_dir: str | os.PathLike,
    model_path: str,
    parquet_path: str,
    num_params: int,
    
    model_name: str,
    layer_counts: tuple,
    
    w0: float ,
    w_hidden: float,
    s: float,
    beta: float,
    k: float,
    global_z: bool ,
    regularize_hyperparams: bool,
    encoder_params: tuple,
    lr: float,
    label_mode: str
):
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    save_path = out_path / model_path
    
    log_path = pathlib.Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    csv_path = log_path / (save_path.stem + ".csv")
    
    hyper_params = [
        f"ω0={trimf(w0)}",
        f"ωh={trimf(w_hidden)}",
        f"s={trimf(s)}",
        f"β={trimf(beta)}",
        f"k={trimf(k)}",
        f"z={'global' if global_z else 'local'}",
        f"reg={regularize_hyperparams}",
        
    ]
    enc_params = [
        f"m={trimf(encoder_params[0])}",
        f"σ={trimf(encoder_params[1])}",
        f"α={trimf(encoder_params[2])}",
    ]
    hyper_params = ';'.join(hyper_params)
    enc_params = ';'.join(enc_params)


    # Static Global Params
    global_meta = {
        "model": model_name,
        "layers": pretty_tuple(layer_counts), 
        "mode": label_mode,
        "lr": lr,
        "params": f"{trimf(num_params*1e-6)}M",
        "size": f"{trimf(num_params*4e-6)}MB",
        "hyper_params": hyper_params,
        "encoder_params": enc_params,
        "train_set": pathlib.Path(parquet_path).name.replace('.parquet', ''),
    }
    
    return global_meta, save_path, csv_path

# ===================== PUBLIC API =====================

def train_and_eval(
    parquet_path: str,
    codes_path: str | None = COUNTRIES_ECOC_PATH,
    out_dir: str | os.PathLike = CHECKPOINT_PATH,
    log_dir: str | os.PathLike = TRAINING_LOG_PATH,
    
    batch_size: int = 8192,
    epochs: int = 10,
    
    model_name: str = "siren",
    layer_counts: tuple = (256,)*5,
    
    w0: float = 30.0,
    w_hidden: float = 1.0,
    s: float = 1.0,
    beta: float = 1.0,
    k: float =  20.0,
    global_z: bool = True,
    regularize_hyperparams: bool = False,
    
    encoder_params: tuple = (16, 2.0 * math.pi, 1.0),
    
    lr: float = 9e-4,
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_uncertainty_loss_weighting: bool = False,
    
    val_lambda = 50,
    
    label_mode: str = "ecoc",
    
    device: str | None = None,
    head_layers: tuple = ()
):
    """
    Trains and evaluates a NIR architecture, saving the best out of all into a checkpoint.
    """
    if device is None:
        device = get_default_device()
    print(f"Training on device: {device}")

    t0 = time.perf_counter()
    # 1. Build Model
    model, model_path = build_model(
        model_name, 
        layer_counts,
        label_mode,
        (w0, w_hidden, s, beta, global_z),
        encoder_params, 
        regularize_hyperparams=regularize_hyperparams)
    model = model.to(device)

    # 2. Uncertainty Weighting
    uw = UncertaintyWeighting().to(device) if use_uncertainty_loss_weighting else None

    # 3. Data Loaders
    train_loader, val_loader, class_cfg, codebook = make_dataloaders(
        parquet_path=parquet_path,
        label_mode=label_mode,
        codes_path=codes_path,
        split=(0.9, 0.1),
        batch_size=batch_size,
        codebook=None,
        device=device
    )

    # 4. Optimizer
    params = [{"params": model.parameters()}]
    if uw is not None:
        params.append({"params": uw.parameters(), "weight_decay": 0.0})
    
    opt = torch.optim.AdamW(params, lr=lr)
    
    # 5. ECOC pos Weights 
    pw_c1, pw_c2 = compute_potential_ecoc_pos_weights(parquet_path, codebook, label_mode)
    
    # 6. Trainer Instance
    trainer = Trainer(
        model=model,
        model_name=model_name,
        optimizer=opt,
        device=device,
        loss_weights=LossWeights(*loss_weights),
        class_cfg=class_cfg,
        pos_weight_c1=pw_c1,
        pos_weight_c2=pw_c2,
        codebook=codebook,
        uw=uw,
        regularize_hyperparams=regularize_hyperparams
    )

    # 7. Logging Setup
    num_params = sum(p.numel() for p in model.parameters())
    
    global_meta, save_path, csv_path = setup_logging(
        out_dir,log_dir,model_path,parquet_path,num_params,model_name,layer_counts,
        w0,w_hidden,s,beta,k,global_z,regularize_hyperparams,encoder_params,lr,label_mode)

    history = []
    best_score = math.inf
    
    dt = time.perf_counter() - t0
    print(f"Total Setup Time: {dt:.3f}s")
    start_time = time.perf_counter()

    
    print(f"Start Training: {model_name} | {pretty_tuple(layer_counts)} | {label_mode}")
    
    # 8. Training Loop
    for ep in range(1, epochs + 1):
        t0 = time.perf_counter()
        tr_stats = trainer.train_epoch(train_loader)
        val_stats = trainer.eval_epoch(val_loader)
        epoch_time = time.perf_counter() - t0
        
        # --- Checkpoint Score ---
        rmse = val_stats.get("glob_RMSE")
        acc_c1 = val_stats.get("c1_BalAcc", val_stats.get("c1_DecAcc", 0.0))
        acc_c2 = val_stats.get("c2_BalAcc", val_stats.get("c2_DecAcc", 0.0))
        
        val_score = rmse + val_lambda * ((1.0 - acc_c1) + (1.0 - acc_c2))
        
        # --- Record ---
        row = {
            "epoch": ep, 
            "time_ep": epoch_time, 
            "val_score": val_score,
            **global_meta
        }
        # Flatten stats with prefixes
        for k,v in tr_stats.items(): row[f"train_{k}"] = v
        for k,v in val_stats.items(): row[f"val_{k}"] = v
        
        history.append(row)
        # Incremental Save (Prevents data loss if crash)
        pd.DataFrame(history).to_csv(csv_path, index=False)
        
        # --- Print ---
        # Concise print
        print(f"[{ep:02d}] T:{tr_stats['loss']} | V_RMSE:{trimf(rmse)} | V_BalAcc:{trimf(acc_c1,2)}/{trimf(acc_c2,2)} | Score:{trimf(val_score)}")

        # --- Save ---
        if val_score < best_score:
            best_score = val_score
            print(f"    ★ Saved Best: {save_path.name}")
            torch.save({
                "model": model.state_dict(),
                "config": {
                    **global_meta,
                    "n_bits": getattr(class_cfg, "n_bits", None),
                    "n_classes_c1": getattr(class_cfg, "n_classes_c1", None),
                    "n_classes_c2": getattr(class_cfg, "n_classes_c2", None),
                    "pos_weight_c1": pw_c1.cpu().tolist() if pw_c1 is not None else None,
                    "pos_weight_c2": pw_c2.cpu().tolist() if pw_c2 is not None else None,
                    "uw": uw.state_dict() if uw is not None else 1.0,
                },
                "epoch": ep
            }, save_path)
    
    total_time = time.perf_counter() - start_time
    print(f"Done. Total Time: {total_time:.1f}s")
            