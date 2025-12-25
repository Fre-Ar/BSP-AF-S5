# src/nirs/metrics.py

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, Optional
from geodata.ecoc.ecoc import ecoc_decode, per_bit_threshold

def compute_distance_metrics(
    pred_km: torch.Tensor, 
    target_km: torch.Tensor, 
    bands: list[tuple[float, float]] = [(0, 25), (25, 150), (150, float('inf'))]
) -> Dict[str, float]:
    """
    Computes distance metrics (RMSE, MedAE, etc.) globally and per-band.
    Inputs should be flat 1D tensors in km.
    """
    # Ensure inputs are on CPU/numpy for quantiles/pandas ops if needed, 
    # but torch is faster for basic stats.
    # We'll stay in Torch for speed, move to float for result.
    
    diff = pred_km - target_km
    abs_diff = diff.abs()
    
    metrics = {}
    
    # --- Helper Inner Function ---
    def _calc(subset_diff, subset_abs, prefix):
        if subset_diff.numel() == 0:
            return
        
        # Basic
        # TODO: make sure that the use of this mse_loss is actually performant.
        metrics[f"{prefix}RMSE"] = F.mse_loss(subset_diff, torch.zeros_like(subset_diff), reduction='mean').sqrt().item()
        metrics[f"{prefix}MAE"] = subset_abs.mean().item()
        metrics[f"{prefix}ME"] = subset_diff.mean().item() # Mean Error (Bias)
        
        # Variance of Error (Error Variance)
        metrics[f"{prefix}VE"] = torch.var(subset_diff).item()
        
        # Percentiles (requires sorting)
        # Median Absolute Error
        metrics[f"{prefix}MedAE"] = subset_abs.median().item()
        # 95th Percentile Absolute Error
        metrics[f"{prefix}P95AE"] = torch.quantile(subset_abs, 0.95).item()

    # 1. Global
    _calc(diff, abs_diff, "glob_")
    
    # 2. Per Band
    for (low, high) in bands:
        mask = (target_km >= low) & (target_km < high)
        if mask.any():
            band_name = f"b{low}-{high if high != float('inf') else 'inf'}_"
            _calc(diff[mask], abs_diff[mask], band_name)
            
    return metrics

def compute_classification_metrics(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    label_mode: str,
    codes_mat: torch.Tensor,         # Pre-converted [C, K] float tensor
    class_ids: torch.Tensor,         # Pre-converted [C] long tensor
    pos_weight: Optional[torch.Tensor] = None,
    bits: Optional[torch.Tensor] = None
) -> Dict[str, float]:
    """
    Computes classification metrics: Top1, BalAcc, ECE, T2Gap, Conf.
    
    logits:  cn_logits
    targets: cn_idx
    bits:    cn_bits
    """
    metrics = {}
    targets = targets.to(logits.device)
    
    if label_mode == "ecoc":
        assert bits is not None, "ECOC mode requires bits"
        assert codes_mat is not None, "ECOC mode requires codebook"
        
        # 1. Compute Bit Accuracy
        bits = bits.to(logits.device)
        # Calculate threshold based on pos_weight 
        threshold = per_bit_threshold(pos_weight, logits.device, bits.size(1))
        pred_bits = (torch.sigmoid(logits) > threshold).float()
        metrics["BitAcc"] = (pred_bits == bits).float().mean().item()

        # 2. Decode to get Class Predictions and Raw Stats
        # All tensors of shape [B]
        idx_pred, confs, gaps = ecoc_decode(
            logits, codes_mat, class_ids, pos_weight, mode='soft', full_return=True)
        num_classes = codes_mat.shape[0]
        
    else:  
        # Standard Softmax: Logits are already class scores
        class_scores = logits
        num_classes = logits.size(1) 
        
        # Probabilities & Top2Gap
        probs = F.softmax(class_scores, dim=1)
        
        # Get Top-2 for Conf and Gap 
        top2_vals, top2_idx = probs.topk(k=2, dim=1)
        
        confs = top2_vals[:, 0]       # Top-1 probability
        gaps  = top2_vals[:, 0] - top2_vals[:, 1] # Margin

        idx_pred = top2_idx[:, 0]

    # Scalar Aggregates (Mean)
    metrics["T2Gap"] = gaps.mean().item()
    metrics["Conf"] = confs.mean().item()
    
    # Compute Expensive Stats
    stats = _compute_vectorized_stats(
        preds = idx_pred,
        targets = targets,
        confs = confs,
        num_classes=num_classes,
        n_bins=10
    )
    metrics.update(stats)
    
    return metrics


def _compute_vectorized_stats(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    confs: torch.Tensor, 
    num_classes: int,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Helper: Computes ECE, DecAcc, and BalAcc using vectorized GPU operations.
    Shared by both ECOC and Softmax modes.
    """
    stats = {}
    device = preds.device
    
    # 1. Decoding Accuracy (Top-1)
    correct = (preds == targets).float()
    stats["DecAcc"] = correct.mean().item()
    
    # 2. Expected Calibration Error (ECE)
    # Assign bins: 0.0-1.0 -> 0..(n_bins-1)
    bins = (confs * n_bins).long().clamp(0, n_bins - 1)
    
    bin_count    = torch.zeros(n_bins, device=device)
    bin_acc_sum  = torch.zeros(n_bins, device=device)
    bin_conf_sum = torch.zeros(n_bins, device=device)
    
    # Scatter Add: Aggregate sums per bin
    bin_count.scatter_add_(0, bins, torch.ones_like(confs))
    bin_acc_sum.scatter_add_(0, bins, correct)
    bin_conf_sum.scatter_add_(0, bins, confs)
    
    # Calculate ECE only on non-empty bins
    mask = bin_count > 0
    if mask.any():
        avg_acc  = bin_acc_sum[mask] / bin_count[mask]
        avg_conf = bin_conf_sum[mask] / bin_count[mask]
        prop_bin = bin_count[mask] / preds.size(0)
        stats["ECE"] = torch.sum(prop_bin * (avg_acc - avg_conf).abs()).item()
    else:
        stats["ECE"] = 0.0

    # 3.Balanced Accuracy
    # Recall = TP / (TP + FN) per class
    # Confusion Matrix diagonals (TP) and Support (TP+FN)
    # We use scatter_add again
    class_correct = torch.zeros(num_classes, device=device)
    class_total   = torch.zeros(num_classes, device=device)
    
    # targets must be 0..C-1
    class_total.scatter_add_(0, targets, torch.ones_like(targets, dtype=torch.float))
    class_correct.scatter_add_(0, targets, correct)
    
    # Mean Recall of classes present in the batch
    mask = class_total > 0
    if mask.any():
        recalls = class_correct[mask] / class_total[mask]
        stats["BalAcc"] = recalls.mean().item()
    else:
        stats["BalAcc"] = 0.0
        
    return stats
