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
    Computes regression metrics for distance estimation, calculated both globally 
    and within specific spatial bands.

    Metrics calculated:
    - **RMSE**: Root Mean Square Error.
    - **MAE**: Mean Absolute Error.
    - **ME**: Mean Error (Bias).
    - **VE**: Variance of Error.
    - **MedAE**: Median Absolute Error (Robust to outliers).
    - **P95AE**: 95th Percentile Absolute Error (Worst-case performance).

    Parameters
    ----------
    pred_km : torch.Tensor
        1D Tensor of predicted distances in kilometers.
    target_km : torch.Tensor
        1D Tensor of ground truth distances in kilometers.
    bands : list[tuple[float, float]]
        List of (min, max) distance intervals. Metrics are computed separately 
        for points falling into each band (e.g., "b0-25_RMSE").

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary mapping metric names (e.g., "glob_RMSE", "b0-25_MAE") to values.
    """
    
    diff = pred_km - target_km
    abs_diff = diff.abs()
    
    metrics = {}
    
    # --- Helper Inner Function ---
    def _calc(subset_diff, subset_abs, prefix):
        if subset_diff.numel() == 0:
            return
        
        # Basic
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
    target_dist_km: torch.Tensor,
    label_mode: str,
    codes_mat: torch.Tensor,         # Pre-converted [C, K] float tensor
    class_ids: torch.Tensor,         # Pre-converted [C] long tensor
    pos_weight: Optional[torch.Tensor] = None,
    bits: Optional[torch.Tensor] = None,
    bands: list[tuple[float, float]] = [(0, 25), (25, 150), (150, float('inf'))]
) -> Dict[str, float]:
    """
    Computes classification performance metrics (Accuracy, Calibration, Confidence) 
    globally and per distance band. Supports both ECOC and Softmax modes.

    Metrics calculated:
    - **DecAcc**: Decoding Accuracy (Top-1).
    - **BalAcc**: Balanced Accuracy (Mean Recall per class).
    - **ECE**: Expected Calibration Error.
    - **T2Gap**: Gap between Top-1 and Top-2 probabilities (Confidence margin).
    - **Conf**: Top-1 Confidence.
    - **BitAcc**: (ECOC only) Mean accuracy of individual code bits.

    Parameters
    ----------
    logits : torch.Tensor
        Model output. Shape [N, K] where K is num_classes (softmax) or n_bits (ECOC).
    targets : torch.Tensor
        Ground truth labels, aka Country IDs.
        Shape [N].
    target_dist_km : torch.Tensor
        Ground truth distance, used to filter samples into spatial bands. Shape [N].
    label_mode : str
        "ecoc" or "softmax". Determines decoding strategy.
    codes_mat : torch.Tensor
        ECOC codebook matrix [C, K]. Required if label_mode="ecoc".
    class_ids : torch.Tensor
        Mapping from codebook index to raw Country ID [C]. Required for ECOC decoding.
    pos_weight : torch.Tensor or None
        Positive weights for BCE loss, used to calculate decision thresholds in ECOC.
    bits : torch.Tensor or None
        Ground truth bit vectors corresponding to `targets`. Shape [N, n_bits].
        Required if label_mode="ecoc" to compute BitAcc.
    bands : list[tuple[float, float]]
        Distance intervals for stratified evaluation.

    Returns
    -------
    metrics : Dict[str, float]
        Dictionary mapping metric names (e.g., "glob_DecAcc", "b0-25_ECE") to values.
    """
    metrics = {}
    device = logits.device
    targets = targets.to(device)
    target_dist_km = target_dist_km.to(device)
    
    # --- Global Pre-computation (Vectorized) ---
    # We decode everything once to get vectors of length N
    if label_mode == "ecoc":
        assert bits is not None, "ECOC mode requires bits"
        assert codes_mat is not None, "ECOC mode requires codebook"
        
        # 1. Compute Bit Accuracy
        bits = bits.to(device)
        # Calculate threshold based on pos_weight 
        threshold = per_bit_threshold(pos_weight, device, bits.size(1))
        pred_bits = (torch.sigmoid(logits) > threshold).float()
        # Mean bit accuracy per sample
        bit_correct_vec = (pred_bits == bits).float().mean(dim=1)
        
        # 2. Decode to get Class Predictions and Raw Stats
        # All tensors of shape [B]
        id_pred, confs, gaps = ecoc_decode(
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

        # TODO: This idx might not be the indices
        id_pred = top2_idx[:, 0]
        
    # Helper for Aggregation
    def _calc_subset(mask, prefix):
        if not mask.any(): return

        # Scalar aggregates (Mean)
        metrics[f"{prefix}T2Gap"] = gaps[mask].mean().item()
        metrics[f"{prefix}Conf"] = confs[mask].mean().item()
        
        if bit_correct_vec is not None:
            metrics[f"{prefix}BitAcc"] = bit_correct_vec[mask].mean().item()
            
        # Compute Expensive stats (DecAcc, ECE, BalAcc)
        # Note: We pass the filtered tensors
        subset_stats = _compute_vectorized_stats(
            preds=id_pred[mask], 
            targets=targets[mask], 
            confs=confs[mask], 
            num_classes=num_classes,
            n_bins=10
        )
        
        # Add prefix
        for k, v in subset_stats.items():
            metrics[f"{prefix}{k}"] = v

    # --- Compute Global & Bands ---
    
    # Global
    _calc_subset(torch.ones_like(targets, dtype=torch.bool), "glob_")
    
    # Per Band
    for (low, high) in bands:
        mask = (target_dist_km >= low) & (target_dist_km < high)
        if mask.any():
            band_name = f"b{low}-{high if high != float('inf') else 'inf'}_"
            _calc_subset(mask, band_name)
    
    return metrics


def _compute_vectorized_stats(
    preds: torch.Tensor, 
    targets: torch.Tensor, 
    confs: torch.Tensor, 
    num_classes: int,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Helper function to compute expensive aggregate statistics (ECE, BalAcc) 
    using vectorized scatter operations on GPU.

    Parameters
    ----------
    preds : torch.Tensor
        Predicted class ids [N].
    targets : torch.Tensor
        Ground truth class ids [N].
    confs : torch.Tensor
        Confidence (probability) of the predicted class [N].
    num_classes : int
        Total number of classes (for confusion matrix/BalAcc).
    n_bins : int
        Number of bins for Expected Calibration Error (ECE) calculation.

    Returns
    -------
    stats : Dict[str, float]
        Dictionary containing "DecAcc", "ECE", and "BalAcc".
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
