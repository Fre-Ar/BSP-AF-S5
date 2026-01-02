import os
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from .nns.nir import ClassHeadConfig
from geodata.ecoc.ecoc import (
    ecoc_prevalence_by_bit,
    pos_weight_from_prevalence,
)

def compute_class_counts(
    data_dir: str,
    id_col: str = "c1_id",
    num_classes: int = 289,
) -> np.ndarray:
    """
    Iterates over all parquet files in data_dir to count occurrences of `id_col`.
    Returns a numpy array of shape (num_classes,) containing the counts.
    
    Parameters
    ----------
    data_dir : str
        Path to directory containing parquet files.
    id_col : str, optional
        Name of the column containing class ids (e.g. "c1_id" or "c2_id").
    num_classes : int
        Total number of classes.

    Returns
    -------
    ones : ndarray
        (num_classes,) int64, total counts.
    """
    # Initialize global counter
    class_counts = np.zeros(num_classes, dtype=np.int64)
    
    files = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
    if not files:
        raise ValueError(f"No parquet files found in {data_dir}")
        
    print(f"Streaming {len(files)} files to compute global class stats...")
    
    for fpath in tqdm(files):
        # Only load the necessary column
        df = pd.read_parquet(fpath, columns=[id_col])
        
        # Count values in this chunk
        # Assuming id_col values are integers in [0, num_classes-1]
        counts = df[id_col].value_counts()
        
        # Get IDs and frequencies
        ids = counts.index.to_numpy()
        values = counts.values
        
        # Map ID 1 -> Index 0
        indices = ids - 1
        
        # Safety Checks
        if indices.min() < 0:
            invalid_id = ids[indices.argmin()]
            raise ValueError(f"Found ID {invalid_id} <= 0. IDs must be 1-based.")
            
        if indices.max() >= num_classes:
            invalid_id = ids[indices.argmax()]
            raise ValueError(f"Found ID {invalid_id} > num_classes ({num_classes}).")

        class_counts[indices] += values
        # Explicit cleanup
        del df, counts, ids, values, indices
    
    return class_counts

def compute_pos_weights(
    data_dir: str,
    codebook: dict,
    class_cfg: ClassHeadConfig):
    """
    Computes pos_weight for c1/c2 over a batch of parquet files.
    Currently only supports softmax weights.

    Returns
    -------
    pw_c1, pw_c2 : torch.Tensor
    """
    if codebook is None:
        return None, None
    
    if class_cfg.label_mode == "ecoc":
        raise NotImplementedError("ECOC pos_weight batch computation not implemented in this function.")
        # BCE bit prevalence correction
        ones_c1, totals_c1, p1_c1 = ecoc_prevalence_by_bit(parquet_path, codebook, id_col="c1_id")
        ones_c2, totals_c2, p1_c2 = ecoc_prevalence_by_bit(parquet_path, codebook, id_col="c2_id")

        pw_c1 = pos_weight_from_prevalence(p1_c1)
        pw_c2 = pos_weight_from_prevalence(p1_c2)
    else: # Softmax
        # 1. Calculate counts for every class ID
        counts_c1 = compute_class_counts(data_dir, "c1_id", class_cfg.n_classes_c1)
        counts_c2 = compute_class_counts(data_dir, "c2_id", class_cfg.n_classes_c2)

        # 2. Inverse Frequency Weighting
        # w_c = Total / (N_classes * count_c) is a standard balanced heuristic
        # Add epsilon to avoid div by zero
        w_c1 = counts_c1.sum() / (class_cfg.n_classes_c1 * (counts_c1 + 1))
        w_c2 = counts_c2.sum() / (class_cfg.n_classes_c2 * (counts_c2 + 1))
        
        pw_c1 = torch.tensor(w_c1).float()
        pw_c2 = torch.tensor(w_c2).float()

    return pw_c1, pw_c2