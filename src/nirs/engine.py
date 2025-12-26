# src/nirs/engine.py

from __future__ import annotations

from typing import Union
import torch
import numpy as np
from pathlib import Path
import pyarrow.parquet as pq

from geodata.ecoc.ecoc import (
    load_ecoc_codes,
    ecoc_prevalence_by_bit,
    pos_weight_from_prevalence,
    ecoc_decode,
    _prepare_codebook_tensor
)
from nirs.create_nirs import build_model
from utils.utils import get_default_device
from .inference import InferenceConfig, Prediction
from .nns.nir import ClassHeadConfig

def _compute_class_counts(
    parquet_path: str | Path,
    id_col: str = "c1_id",
    n_classes: int = 289,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes prevalence of classes in a parquet file.

    Parameters
    ----------
    parquet_path : str or Path
        Path to Parquet file with a column `id_col` of integer class IDs.
    id_col : str, optional
        Name of the column containing class ids (e.g. "c1_id" or "c2_id").
    n_classes : int
        Total number of classes.

    Returns
    -------
    ones : ndarray
        (n_classes,) int64, total counts.
    """
    parquet_path = Path(parquet_path)
     # init array
    counts = np.zeros(n_classes, dtype=np.int64)

    # read the data
    pf = pq.ParquetFile(str(parquet_path))
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[id_col])
        ids = tbl[id_col].to_numpy(zero_copy_only=False)

        # Bin counts. 
        # minlength ensures we get a vector of at least size n_classes.
        bc = np.bincount(ids, minlength=n_classes)
        # bc will be len=290 if Ids are 1..289
        
        # We slice it back to match the expected model output size.
        # All ids becomes id-1
        bc = bc[1:]

        counts += bc
    
    return counts

def compute_pos_weights(
    parquet_path: str,
    codebook: dict,
    class_cfg: ClassHeadConfig):
    """
    Computes pos_weight for c1/c2 over a full parquet.
    Supports both ECOC per-bit pos_weights and softmax weights.

    Returns
    -------
    pw_c1, pw_c2 : torch.Tensor
    """
    if codebook is None:
        return None, None
    
    if class_cfg.label_mode == "ecoc":
        # BCE bit prevalence correction
        ones_c1, totals_c1, p1_c1 = ecoc_prevalence_by_bit(parquet_path, codebook, id_col="c1_id")
        ones_c2, totals_c2, p1_c2 = ecoc_prevalence_by_bit(parquet_path, codebook, id_col="c2_id")

        pw_c1 = pos_weight_from_prevalence(p1_c1)
        pw_c2 = pos_weight_from_prevalence(p1_c2)
    else: # Softmax
        # 1. Calculate counts for every class ID
        # (You can implement a helper similar to ecoc_prevalence_by_bit but for ints)
        counts_c1 = _compute_class_counts(parquet_path, "c1_id", class_cfg.n_classes_c1)
        counts_c2 = _compute_class_counts(parquet_path, "c2_id", class_cfg.n_classes_c2)
        
        # 2. Inverse Frequency Weighting
        # w_c = Total / (N_classes * count_c) is a standard balanced heuristic
        # Add epsilon to avoid div by zero
        w_c1 = counts_c1.sum() / (class_cfg.n_classes_c1 * (counts_c1 + 1))
        w_c2 = counts_c2.sum() / (class_cfg.n_classes_c2 * (counts_c2 + 1))
        
        pw_c1 = torch.tensor(w_c1).float()
        pw_c2 = torch.tensor(w_c2).float()

    return pw_c1, pw_c2


# -------------------------------------------------------------------
# Learnable hyperparameter inspection (INCODE a,b,c,d sanity check)
# -------------------------------------------------------------------
def summarize_abcd(a, b, c, d, label=""):
    """
    Prints simple summary stats (mean/std/min/max) for four tensors a,b,c,d.

    Useful for checking that INCODE hyperparameters don't explode or collapse.
    """
    def s(t):
        return dict(mean=float(t.mean()), std=float(t.std()),
                    min=float(t.min()), max=float(t.max()))
    print(f"[{label}] a={s(a)}")
    print(f"[{label}] b={s(b)}")
    print(f"[{label}] c={s(c)}")
    print(f"[{label}] d={s(d)}")

@torch.no_grad()
def check_abcd(model, x_batch):
    """
    Runs a single forward pass with `return_abcd=True` and print stats for (a,b,c,d).

    Assumes model(x, return_abcd=True) -> (dist, c1, c2, (a,b,c,d)).
    """
    model.eval()
    if x_batch.dim() == 1: 
        x_batch = x_batch.unsqueeze(0)
    dist, c1, c2, (a,b,c,d) = model(x_batch, return_abcd=True)
    summarize_abcd(a, b, c, d, label=f"B={x_batch.size(0)}")
    return a, b, c, d

class Predictor:
    """
    Wraps a trained NIR model to provide a simple predict() API.
    Handles device management, ECOC decoding, and distance transforms.
    """
    def __init__(self, cfg: InferenceConfig, checkpoint_path: str, device: str | None):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.device = device if device else get_default_device()
        
        print(f"[Predictor] Loading {cfg.model_name} from {checkpoint_path}...")
        
        # 1. Load Model & Codebook
        
        # ---- load checkpoint & config ----
        self.ckpt = torch.load(checkpoint_path, map_location="cpu")
        
        # ---- build & load model ----
        self.model, _ = build_model(model_cfg=cfg)
        self.model.to(self.device)
        self.model.load_state_dict(self.ckpt["model"])
        self.model.eval()

        self.codebook = None
        if cfg.label_mode == "ecoc":
            if cfg.codes_path is None:
                raise ValueError("ECOC mode requires codes_path to the ECOC JSON codebook.")
            self.codebook = load_ecoc_codes(cfg.codes_path)
        
        # 2. Prepare Codebook Tensors (for ECOC decoding)
        if self.cfg.label_mode == "ecoc":
            self.class_ids, self.codes_mat = _prepare_codebook_tensor(
                self.codebook, self.device
            )
            
            # 3. Recover Position Weights from Checkpoint (for Soft ECOC consistency)
            # training.py saves these in the "config" dict inside the checkpoint
            ckpt_cfg = self.ckpt.get("config", {})
            
            self.pw_c1 = None
            if "pos_weight_c1" in ckpt_cfg and ckpt_cfg["pos_weight_c1"] is not None:
                self.pw_c1 = torch.tensor(ckpt_cfg["pos_weight_c1"], device=self.device)
                
            self.pw_c2 = None
            if "pos_weight_c2" in ckpt_cfg and ckpt_cfg["pos_weight_c2"] is not None:
                self.pw_c2 = torch.tensor(ckpt_cfg["pos_weight_c2"], device=self.device)
        else:
            self.class_ids = None
            self.codes_mat = None
            self.pw_c1 = None
            self.pw_c2 = None

    def _forward_model(self, x: torch.Tensor, check_abcd: bool = False):
        """Unified forward pass handling model-specific signatures (e.g. INCODE)."""
        if self.cfg.model_name.lower() == "incode" and check_abcd:
            check_abcd(self.model, x)
        return self.model(x)

    @torch.no_grad()
    def predict(self, xyz: Union[np.ndarray, torch.Tensor], tau: float = 1.0, return_logits: bool = False) -> Prediction:
        """
        Runs inference on a batch of points.
        
        Args:
            xyz: (N, 3) array or tensor of unit vectors.
            tau: Temperature for ECOC decoding (default 1.0).
            return_logits: If True, includes raw logits in the output.
            
        Returns:
            Prediction object with cpu numpy arrays.
        """
        # 1. Prepare Input
        if isinstance(xyz, np.ndarray):
            x_t = torch.from_numpy(xyz).to(self.device, dtype=torch.float32)
        else:
            x_t = xyz.to(self.device, dtype=torch.float32)
        
        # 2. Forward Pass
        pred_log1p, c1_logits, c2_logits = self._forward_model(x_t)
        
        # 3. Distance Transform
        pred_log1p = pred_log1p.squeeze(-1)
        if self.cfg.model_outputs_log1p:
            dist_km = torch.expm1(pred_log1p)
        else:
            dist_km = pred_log1p
            
        # 4. Decoding (Classification)
        if self.cfg.label_mode == "ecoc":
            # Apply temperature scaling
            l_c1 = c1_logits / tau
            l_c2 = c2_logits / tau
            
            c1_ids = ecoc_decode(l_c1, self.codes_mat, self.class_ids, self.pw_c1, mode="soft")
            c2_ids = ecoc_decode(l_c2, self.codes_mat, self.class_ids, self.pw_c2, mode="soft")
        else:
            # Softmax Argmax
            # Logits are indices [0..288]. Dataset IDs are [1..289].
            # We MUST shift by +1 to return Raw IDs.
            c1_ids = c1_logits.argmax(dim=1) + 1
            c2_ids = c2_logits.argmax(dim=1) + 1
            
        return Prediction(
            dist_km=dist_km.cpu().numpy(),
            c1_ids=c1_ids.long().cpu().numpy(),
            c2_ids=c2_ids.long().cpu().numpy(),
            logits_c1=c1_logits.cpu().numpy() if return_logits else None,
            logits_c2=c2_logits.cpu().numpy() if return_logits else None
        )