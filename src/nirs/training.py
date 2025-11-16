# src/nirs/training.py

import math, pathlib
import pandas as pd, numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Literal
from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F
import torch.nn as nn, torch.optim as opt
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


from .loss import UncertaintyWeighting
from .nns.nn_incode import INCODE_NIR as Incode
from .nns.nir import LabelMode
from .create_nirs import build_model

from geodata.ecoc.ecoc import per_bit_threshold, ecoc_decode
from utils.utils import get_default_device
from utils.utils_geo import (
    COUNTRIES_ECOC_PATH, 
    CHECKPOINT_PATH,
    SEED)

from .engine import (
    make_dataloaders,
    compute_potential_ecoc_pos_weights
)

# ===================== DATA =====================


class BordersParquet(Dataset):
    """
    Parquet-backed dataset for training the spherical distance + country classification heads.

    Expected Parquet columns:
      - lon, lat          : geographic coordinates in degrees (unused at training time if dropped)
      - x, y, z           : unit-vector coordinates on the sphere
      - dist_km           : scalar geodesic distance to nearest border segment (km)
      - log1p_dist        : log(1 + dist_km)
      - c1_id, c2_id      : integer class IDs for the two “sides” of the nearest border
      - is_border         : 1 if sampled via near-border process, 0 if uniform globe
      - r_band            : distance band index (0..N for near-border, 255 for uniform)

    Label modes
    -----------
    ECOC mode (label_mode="ecoc"):
      - Requires `codebook: Dict[int, ndarray]` mapping class_id -> bit vector (0/1) of shape (n_bits,).
      - __getitem__ returns:
          {
            "xyz":        FloatTensor (3,),
            "dist":       FloatTensor (1,),        # km
            "log1p_dist": FloatTensor (1,),
            "r_band":     IntTensor (1,),
            "c1_idx":     LongTensor (),           # raw class id
            "c2_idx":     LongTensor (),
            "c1_bits":    FloatTensor (n_bits,),
            "c2_bits":    FloatTensor (n_bits,),
          }

    Softmax mode (label_mode="softmax"):
      - Ignores `codebook`.
      - __getitem__ returns the same keys minus "c1_bits"/"c2_bits".

    Splitting
    ---------
    - Deterministic shuffle with `seed`.
    - `split_frac = (train_frac, val_frac)` must sum to 1.
    - `split="train"` → first train_frac of shuffled indices.
      `split="val"`   → remaining indices.

    Notes
    -----
    - All features/targets are preloaded into memory as tensors for fast __getitem__.
      This is fine for up to ~10M rows on a modern machine; for larger datasets you
      might want streaming / lazy loading.
    """
    def __init__(
        self,
        parquet_path: str | pathlib.Path,
        
        split: Literal["train", "val"] = "train",
        split_frac: Tuple[float, float] = (0.9, 0.1),
        
        seed: int = SEED,
        
        codebook:  Optional[Dict[int, np.ndarray]] = None,
        label_mode: LabelMode = "ecoc",
        
        drop_cols: Tuple[str, ...] = ("lon", "lat", "is_border")
    ):
        super().__init__()
        assert abs(sum(split_frac) - 1.0) < 1e-6, "split_frac must sum to 1"
        assert split in ("train", "val")

        self.path = str(parquet_path)
        self.label_mode = label_mode
        # stored for reference; not used inside __getitem__.
        self.codebook = codebook

        # --- load dataframe (pyarrow backend can memory-map) ---
        df = pd.read_parquet(self.path)

        required = {"x", "y", "z", "dist_km", "log1p_dist", "c1_id", "c2_id", "r_band"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Parquet is missing required columns: {sorted(missing)}")

        # --- deterministic shuffle & split ---
        rng = np.random.default_rng(seed)
        idx = np.arange(len(df))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * split_frac[0])
        take = idx[:n_train] if split == "train" else idx[n_train:]
        self.df = df.iloc[take].reset_index(drop=True)

        # keep raw ids for softmax or ECOC lookup  
        c1_id = self.df["c1_id"].to_numpy(np.int32, copy=False)
        c2_id = self.df["c2_id"].to_numpy(np.int32, copy=False)
        
        # --- precompute tensors (fast __getitem__) ---
        # coordinates and regression target
        self.xyz = torch.from_numpy(self.df[["x","y","z"]].values).float()           # (N,3)
        self.dist = torch.from_numpy(self.df["dist_km"].values).float().unsqueeze(1) # (N,1)
        self.log1p_dist = torch.from_numpy(self.df["log1p_dist"].values).float().unsqueeze(1) # (N,1)
        self.r_band = torch.from_numpy(self.df["r_band"].values).int().unsqueeze(1) # (N,1)
        
        # Pre-compute ECOC tensors if applicable
        if label_mode == "ecoc":
            if codebook is None:
                raise ValueError("label_mode='ecoc' requires a codebook.")
            c1_bits = np.stack([codebook[int(cid)] for cid in c1_id], axis=0).astype(np.float32)
            c2_bits = np.stack([codebook[int(cid)] for cid in c2_id], axis=0).astype(np.float32)
            self.c1_bits = torch.from_numpy(c1_bits)
            self.c2_bits = torch.from_numpy(c2_bits)
        else:
            self.c1_bits = None
            self.c2_bits = None
            
        # Drop unused columns to free memory
        for c in drop_cols:
            if c in self.df:
                self.df = self.df.drop(columns=[c])
        
        # Class counts for softmax convenience (assume 0..max_id is dense
        # TODO: fix softmax classes having no 0 index (and thus remove the +1 here).
        self.num_classes_c1 = int(c1_id.max()) + 1
        self.num_classes_c2 = int(c2_id.max()) + 1
        
        self.c1_id = torch.as_tensor(c1_id, dtype=torch.long)
        self.c2_id = torch.as_tensor(c2_id, dtype=torch.long)

    def __len__(self) -> int:
        return self.xyz.shape[0]

    def __getitem__(self, i: int) -> dict:
        item = {
            "xyz":       self.xyz[i],        # (3,)
            "dist":      self.dist[i],       # (1,)
            "log1p_dist":self.log1p_dist[i],       # (1,)
            "r_band":    self.r_band[i],
            "c1_idx":    self.c1_id[i],
            "c2_idx":    self.c2_id[i],
        }
        if self.label_mode == "ecoc":
            item["c1_bits"] = self.c1_bits[i]  # (n_bits,)
            item["c2_bits"] = self.c2_bits[i]  # (n_bits,)
        return item

# ===================== INTERNAL HELPERS =====================


def _accumulate_distance_metrics(
    totals: dict,
    pred_log1p_dist: torch.Tensor,
    log1p_dist: torch.Tensor,
    weight: float,
) -> None:
    """
    Updates running sums for distance metrics in-place.

    totals["mse_sum_log"] accumulates weighted MSE in log1p space,
    totals["mse_sum"]     accumulates weighted MSE in km space.
    """
    totals["mse_sum_log"] += weight * F.mse_loss(
        pred_log1p_dist, log1p_dist, reduction="sum").float()
    totals["mse_sum"] += weight * F.mse_loss(
        torch.expm1(pred_log1p_dist), torch.expm1(log1p_dist), reduction="sum").float()


def _forward_model(
    model: nn.Module,
    xyz: torch.Tensor,
    model_name: str,
    regularize_hyperparams: bool,
):
    """
    Unified forward pass for all NIR models.

    For INCODE, returns:
      (pred_log1p_dist, c1_logits, c2_logits, (a, b, c, d))

    For all other models, returns:
      (pred_log1p_dist, c1_logits, c2_logits, None)
    """
    # INCODE has a slightly different signature and returns extra hyperparams
    if model_name.lower() == "incode":
        pred_log1p_dist, c1_logits, c2_logits, reg_params = model(
            xyz, regularize_hyperparams
        )
    else:
        pred_log1p_dist, c1_logits, c2_logits = model(xyz)
        reg_params = None
    return pred_log1p_dist, c1_logits, c2_logits, reg_params


def _classification_loss(
    c1_logits: torch.Tensor,
    c2_logits: torch.Tensor,
    batch: dict,
    device: torch.device | str,
    label_mode: LabelMode,
    debug_losses: bool,
    # loss modules for non-debug path (can be None if debug)
    bce_c1: nn.Module | None,
    bce_c2: nn.Module | None,
    ce: nn.Module | None,
    pos_weight_c1: Tensor | None,
    pos_weight_c2: Tensor | None,
):
    """
    Computes classification losses for both heads, handling:

      - ECOC vs softmax label mode,
      - debug vs non-debug behavior.

    Returns
    -------
    loss_c1, loss_c2 : scalar tensors
    c1_loss_vec, c2_loss_vec : per-sample losses or None (if not debug)
    """
    if label_mode == "ecoc":
        c1_bits = batch["c1_bits"].to(device)
        c2_bits = batch["c2_bits"].to(device)

        if debug_losses:
            # elementwise BCE: (B, K) -> per-sample mean (B,)
            c1_elem = F.binary_cross_entropy_with_logits(
                c1_logits, c1_bits, reduction="none", pos_weight=pos_weight_c1)
            c2_elem = F.binary_cross_entropy_with_logits(
                c2_logits, c2_bits, reduction="none", pos_weight=pos_weight_c2)
            c1_loss_vec = c1_elem.mean(dim=1) # per-sample
            c2_loss_vec = c2_elem.mean(dim=1) # per-sample

            # scalar losses used for optimization
            loss_c1 = c1_loss_vec.mean()
            loss_c2 = c2_loss_vec.mean()
        else:
            # Use pre-built BCEWithLogitsLoss modules
            assert bce_c1 is not None and bce_c2 is not None
            loss_c1 = bce_c1(c1_logits, c1_bits)
            loss_c2 = bce_c2(c2_logits, c2_bits)
            c1_loss_vec = None
            c2_loss_vec = None

    else:  # softmax
        c1_idx = batch["c1_idx"].to(device)
        c2_idx = batch["c2_idx"].to(device)

        if debug_losses:
            c1_loss_vec = F.cross_entropy(c1_logits, c1_idx, reduction="none")
            c2_loss_vec = F.cross_entropy(c2_logits, c2_idx, reduction="none")

            # scalar losses used for optimization
            loss_c1 = c1_loss_vec.mean()
            loss_c2 = c2_loss_vec.mean()
        else:
            assert ce is not None
            loss_c1 = ce(c1_logits, c1_idx)
            loss_c2 = ce(c2_logits, c2_idx)
            c1_loss_vec = None
            c2_loss_vec = None

    return loss_c1, loss_c2, c1_loss_vec, c2_loss_vec


def _update_ecoc_metrics(
    totals: dict,
    c1_logits: torch.Tensor,
    c2_logits: torch.Tensor,
    batch: dict,
    device: torch.device | str,
    pos_weight_c1: Tensor | None,
    pos_weight_c2: Tensor | None,
    codebook: Optional[Dict[int, np.ndarray]],
):
    """
    Updates ECOC-related validation metrics:

      - bit-level accuracy (c1_bit_acc / c2_bit_acc),
      - decoded top-1 accuracy if codebook + class indices are available.
    """
    c1_bits = batch["c1_bits"].to(device)
    c2_bits = batch["c2_bits"].to(device)

    # Per-bit thresholds consistent with BCE pos_weight
    thr1 = per_bit_threshold(pos_weight_c1, device, c1_bits.size(1))
    thr2 = per_bit_threshold(pos_weight_c2, device, c2_bits.size(1))

    p1 = torch.sigmoid(c1_logits)
    p2 = torch.sigmoid(c2_logits)
    c1_pred_bits = (p1 > thr1).float()
    c2_pred_bits = (p2 > thr2).float()

    # Bit Accuracy
    totals["c1_bits_correct"] += float((c1_pred_bits.eq(c1_bits)).float().sum().item())
    totals["c2_bits_correct"] += float((c2_pred_bits.eq(c2_bits)).float().sum().item())
    totals["n_bits_total"] += int(c1_bits.numel() + c2_bits.numel())

    # Decoded top-1 ECOC accuracy (requires codebook and true indices)
    if codebook is not None:
        c1_idx_true = batch.get("c1_idx")
        c2_idx_true = batch.get("c2_idx")
        if c1_idx_true is not None and c2_idx_true is not None:
            
            c1_idx_true = c1_idx_true.to(device)
            c2_idx_true = c2_idx_true.to(device)

            c1_idx_pred = ecoc_decode(c1_logits, codebook, pos_weight=pos_weight_c1, mode="soft")
            c2_idx_pred = ecoc_decode(c2_logits, codebook, pos_weight=pos_weight_c2, mode="soft")

            totals["c1_decoded_top1"] += (c1_idx_pred == c1_idx_true).float().sum().item()
            totals["c2_decoded_top1"] += (c2_idx_pred == c2_idx_true).float().sum().item()

def _update_softmax_metrics(
    totals: dict,
    c1_logits: torch.Tensor,
    c2_logits: torch.Tensor,
    batch: dict,
    device: torch.device | str,
):
    """
    Updates softmax-based top-1 accuracy metrics.
    """
    c1_idx = batch["c1_idx"].to(device)
    c2_idx = batch["c2_idx"].to(device)

    c1_top1 = c1_logits.argmax(dim=1).eq(c1_idx).float().sum().item()
    c2_top1 = c2_logits.argmax(dim=1).eq(c2_idx).float().sum().item()
    totals["c1_top1"] += float(c1_top1)
    totals["c2_top1"] += float(c2_top1)

# ===================== TRAINING =====================

@dataclass
class LossWeights:
    w_dist: float = 1.0
    w_c1: float = 1.0
    w_c2: float = 1.0

def train_one_epoch(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    
    device: torch.device | str,
    
    lw: LossWeights,
    pos_weight_c1: Tensor | None = None,
    pos_weight_c2: Tensor | None = None,
    
    label_mode: LabelMode = "ecoc",
    
    uw: UncertaintyWeighting | None = None,
    
    debug_losses: bool = False,
    regularize_hyperparams: bool = False
) -> dict:
    """
    Train `model` for one epoch over `loader`.

    Expected batch format
    ---------------------
    Each batch is a dict with at least:
      - "xyz":        FloatTensor [B, 3]
      - "log1p_dist": FloatTensor [B, 1]
      - ECOC mode:
          "c1_bits": FloatTensor [B, K]
          "c2_bits": FloatTensor [B, K]
        Softmax mode:
          "c1_idx":  LongTensor [B]
          "c2_idx":  LongTensor [B]

    Model output:
      model(xyz) -> (pred_log1p_dist, c1_logits, c2_logits)
      where:
        pred_log1p_dist : [B, 1]
        c1_logits, c2_logits : [B, K] for ECOC, or [B, n_classes] for softmax.

    Loss
    ----
      L_dist = MSE(pred_log1p_dist, log1p_dist)
      L_c1   = BCEWithLogits / CE on first head
      L_c2   = BCEWithLogits / CE on second head

      If `uw` is provided (UncertaintyWeighting), the combined loss is:
        uw([w_dist * L_dist, w_c1 * L_c1, w_c2 * L_c2])
      otherwise:
        w_dist * L_dist + w_c1 * L_c1 + w_c2 * L_c2

      For INCODE models, an additional regularization term
      `Incode.incode_reg(a, b, c, d)` is added when `regularize_hyperparams=True`.

    Returns
    -------
    stats : dict
      {
        "loss":        mean total loss,
        "rmse_km":     RMSE in km,
        "rmse_log1p":  RMSE in log1p space,
        "c1_loss":     mean classification loss on head 1,
        "c2_loss":     mean classification loss on head 2,
        # plus debug quantiles (c1_p95/c1_max/c2_p95/c2_max) if debug_losses=True
      }
    """
    # -----------------------------------------------------
    # 1) Initialization
    # -----------------------------------------------------
    model.train()
    totals = {
        "loss": 0.0,
        "mse_sum_log": 0.0,
        "mse_sum": 0.0,
        "n": 0,
        "c1_loss": 0.0,
        "c2_loss": 0.0,
    }
    
    mse = nn.MSELoss(reduction="mean").to(device)
    
    if debug_losses:
        # Track per-sample classification losses for quantiles
        c1_losses_epoch = []
        c2_losses_epoch = []
    else:
        # Standard batched losses
        bce_c1 = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight_c1).to(device)
        bce_c2 = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=pos_weight_c2).to(device)
        ce = nn.CrossEntropyLoss(reduction="mean").to(device)
    
    # -----------------------------------------------------
    # 2) Training
    # -----------------------------------------------------
    pbar = tqdm(loader, leave=False) # progress bar
    for batch in pbar:
        # 2.1) Gradient Descent
        xyz       = batch["xyz"].to(device)           # (B,3)
        log1p_dist= batch["log1p_dist"].to(device)    # (B,1)
        
        # TODO: MAKE r_band MATTER

        optimizer.zero_grad(set_to_none=True)
        
        # forward model
        pred_log1p_dist, c1_logits, c2_logits, reg_params = _forward_model(
            model, xyz, model_name, regularize_hyperparams
        )
        
        # Distance regression in log1p space
        loss_dist = mse(pred_log1p_dist, log1p_dist)
        
        # Classification loss (ECOC vs softmax, debug vs non-debug)
        loss_c1, loss_c2, c1_loss_vec, c2_loss_vec = _classification_loss(
            c1_logits,
            c2_logits,
            batch,
            device,
            label_mode,
            debug_losses,
            bce_c1,
            bce_c2,
            ce,
            pos_weight_c1,
            pos_weight_c2,
        )
        
        # Combine losses (with optional uncertainty weighting)
        if uw:
            loss = uw([lw.w_dist * loss_dist, lw.w_c1 * loss_c1, lw.w_c2 * loss_c2])
        else:
            loss = lw.w_dist * loss_dist + lw.w_c1 * loss_c1 + lw.w_c2 * loss_c2
        
        # Optional INCODE hyperparam regularization
        if model_name.lower() == "incode" and regularize_hyperparams:
            a, b, c, d = reg_params
            loss += Incode.incode_reg(a,b,c,d)
        
        # gradient descent step
        loss.backward()
        optimizer.step()

        # 2.2) Accumulate Stats
        with torch.no_grad():
            B = xyz.size(0)
            totals["loss"] += float(loss.detach()) * B
            totals["mse_sum_log"] += F.mse_loss(pred_log1p_dist.detach(), log1p_dist, reduction="sum").float()
            totals["mse_sum"] += F.mse_loss(torch.expm1(pred_log1p_dist.detach()), torch.expm1(log1p_dist), reduction="sum").float()
            totals["c1_loss"] += float(loss_c1.detach()) * B
            totals["c2_loss"] += float(loss_c2.detach()) * B
            totals["n"] += B
            
            # bit accuracies
            #c1_pred_bits = (torch.sigmoid(c1_logits) >= 0.5).float()
            #c2_pred_bits = (torch.sigmoid(c2_logits) >= 0.5).float()
            #running["bits1"] += (c1_pred_bits.eq(c1_bits)).float().sum().item()
            #running["bits2"] += (c2_pred_bits.eq(c2_bits)).float().sum().item()
            
            if debug_losses:
                 # accumulate per-sample classification losses for quantiles (store on CPU)
                c1_losses_epoch.append(c1_loss_vec.detach().cpu())
                c2_losses_epoch.append(c2_loss_vec.detach().cpu())

            pbar.set_postfix({
                "rmse(km)": (totals["mse_sum"]/totals["n"])**0.5,
                "rmse(log1p)": (totals["mse_sum_log"]/totals["n"])**0.5,
                "c1_loss/n": totals["c1_loss"] / max(1, totals["n"]),
                "c2_loss/n": totals["c2_loss"] / max(1, totals["n"]),
            })
            
        with torch.no_grad():
            # accumulate stats
            B = xyz.size(0)
            totals["loss"] += float(loss.detach()) * B
            _accumulate_distance_metrics(totals, pred_log1p_dist.detach(), log1p_dist, lw.w_dist)
            totals["c1_loss"] += float(loss_c1.detach()) * B
            totals["c2_loss"] += float(loss_c2.detach()) * B
            totals["n"] += B

            if debug_losses and c1_loss_vec is not None and c2_loss_vec is not None:
                # accumulate per-sample classification losses for quantiles (stored on CPU)
                c1_losses_epoch.append(c1_loss_vec.detach().cpu())
                c2_losses_epoch.append(c2_loss_vec.detach().cpu())

            # progress bar printing
            pbar.set_postfix({
                "rmse(km)":    (totals["mse_sum"] / max(1, totals["n"])) ** 0.5,
                "rmse(log1p)": (totals["mse_sum_log"] / max(1, totals["n"])) ** 0.5,
                "c1_loss/n":   totals["c1_loss"] / max(1, totals["n"]),
                "c2_loss/n":   totals["c2_loss"] / max(1, totals["n"]),
            })
    
    # debug_loss stats 
    stats = {}
    if debug_losses:
        # Concatenate and compute quantiles/max
        c1_all = torch.cat(c1_losses_epoch) if len(c1_losses_epoch) else torch.tensor([])
        c2_all = torch.cat(c2_losses_epoch) if len(c2_losses_epoch) else torch.tensor([])

        if c1_all.numel() > 0:
            stats["c1_p95"] = float(torch.quantile(c1_all, 0.95))
            stats["c1_max"] = float(c1_all.max())
        if c2_all.numel() > 0:
            stats["c2_p95"] = float(torch.quantile(c2_all, 0.95))
            stats["c2_max"] = float(c2_all.max())
        if stats:
            print({k: round(v, 6) for k, v in stats.items()})
    
    # return stats
    n = max(1, totals["n"])
    return {
        "loss":      totals["loss"] / n,
        "rmse_km":   math.sqrt(totals["mse_sum"] / n),
        "rmse_log1p":math.sqrt(totals["mse_sum_log"] / n),
        "c1_loss":   totals["c1_loss"] / n,
        "c2_loss":   totals["c2_loss"] / n,
        **stats,
    }

# ===================== EVALUATION =====================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device | str,
    lw: LossWeights,
    pos_weight_c1: Tensor | None = None,
    pos_weight_c2: Tensor | None = None,
    label_mode: LabelMode = "ecoc",
    codebook: Optional[Dict[int, np.ndarray]] = None,
    uw: UncertaintyWeighting | None = None # unused, kept for API compatability
):
    """
    Validation loop (no grad).

    Expected batch format and model outputs are the same as in `train_one_epoch`.

    Metrics
    -------
    Always:
      - "rmse_km"      : RMSE in km (using lw.w_dist)
      - "rmse_log1p"   : RMSE in log1p space (using lw.w_dist)

    If label_mode == "ecoc":
      - "c1_bit_acc"   : fraction of correct bits for c1 head
      - "c2_bit_acc"   : fraction of correct bits for c2 head
      - "c1_decoded_acc", "c2_decoded_acc" if `codebook` is provided and
        the batch includes "c1_idx"/"c2_idx" (decoded ECOC accuracy via `ecoc_decode`).

    If label_mode == "softmax":
      - "c1_top1", "c2_top1" : top-1 accuracy for each classification head.

    Notes
    -----
    - The `uw` argument is currently unused in evaluation; validation is purely
      based on the underlying per-head losses and ECOC decoding logic.
    """
    
    # -----------------------------------------------------
    # 1) Initialization
    # -----------------------------------------------------
    model.eval()
    totals = {
        "mse_sum": 0.0,
        "mse_sum_log": 0.0,
        "n": 0,
        "c1_bits_correct": 0.0,
        "c2_bits_correct": 0.0,
        "n_bits_total": 0,
        "c1_top1": 0.0,
        "c2_top1": 0.0,
        "c1_decoded_top1": 0.0,
        "c2_decoded_top1": 0.0,
    }

    with torch.no_grad():
        for batch in loader:
            xyz     = batch["xyz"].to(device)
            log1p_dist    = batch["log1p_dist"].to(device)

            # forward model
            pred_log1p_dist, c1_logits, c2_logits, _ = _forward_model(
                model, xyz, model_name="siren", regularize_hyperparams=False
            )
            
            # Distance Regression
            _accumulate_distance_metrics(totals, pred_log1p_dist, log1p_dist, lw.w_dist)
            B = xyz.shape[0]
            totals["n"] += B

            # Classification metrics
            if label_mode == "ecoc":
                _update_ecoc_metrics(
                    totals,
                    c1_logits,
                    c2_logits,
                    batch,
                    device,
                    pos_weight_c1,
                    pos_weight_c2,
                    codebook)
            else:
                _update_softmax_metrics(totals, c1_logits, c2_logits, batch, device)

    # stats constructing and returning
    n = max(1, totals["n"])
    out = {
        "rmse_km": math.sqrt(totals["mse_sum"] / n),
        "rmse_log1p": math.sqrt(totals["mse_sum_log"] / n),
    }
    
    if label_mode == "ecoc":
        if totals["n_bits_total"] > 0:
            out["c1_bit_acc"] = totals["c1_bits_correct"] / (totals["n_bits_total"] / 2)
            out["c2_bit_acc"] = totals["c2_bits_correct"] / (totals["n_bits_total"] / 2)
        if codebook is not None and n > 0 and totals["c1_decoded_top1"] + totals["c2_decoded_top1"] > 0:
            out["c1_decoded_acc"] = totals["c1_decoded_top1"] / n
            out["c2_decoded_acc"] = totals["c2_decoded_top1"] / n
    
    else:
        if n > 0:
            out["c1_top1"] = totals["c1_top1"] / n
            out["c2_top1"] = totals["c2_top1"] / n
    return out

# ===================== PUBLIC API =====================

def train_and_eval(
    parquet_path: str,
    codes_path: str | None = COUNTRIES_ECOC_PATH,
    out_dir: str | os.PathLike = CHECKPOINT_PATH,
    
    batch_size: int = 8192,
    epochs: int = 10,
    
    model_name: str = "siren",
    layer_counts: tuple = (256,)*5,
    
    w0: float = 30.0,
    w_hidden: float = 1.0,
    s: float = 1.0,
    beta: float = 1.0,
    global_z: bool = True,
    regularize_hyperparams: bool = False,
    
    encoder_params: tuple = (16, 2.0 * math.pi, 1.0),
    
    lr: float = 9e-4,
    loss_weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    use_uncertainty_loss_weighting: bool = False,
    
    label_mode: str = "ecoc",
    
    device: str | None = None,
    
    debug_losses: bool = False,
    head_layers: tuple = (),
):
    """
    Trains and evaluates a NIR architecture, saving the best out of all into a checkpoint.
    """
    device = get_default_device()
    print(f"Training on device: {device}")

    # ------------------ Model ------------------
    model, model_path = build_model(
        model_name, 
        layer_counts,
        label_mode,
        (w0, w_hidden, s, beta, global_z),
        encoder_params)
    model = model.to(device)

    uw = UncertaintyWeighting().to(device) if use_uncertainty_loss_weighting else None

    # ------------------ Data ------------------
    train_loader, val_loader, class_cfg, codebook = make_dataloaders(
        parquet_path=parquet_path,
        label_mode=label_mode,
        codes_path=codes_path,
        split=(0.9, 0.1),
        batch_size=batch_size,
    )

    # ------------------ Optimizer ------------------
    if uw is not None:
        opt = torch.optim.AdamW(
            [
                {"params": model.parameters()},
                {"params": uw.parameters(), "weight_decay": 0.0},
            ],
            lr=lr,
        )
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)

    lw = LossWeights(*loss_weights)

    # ------------------ ECOC pos_weights ------------------
    pw_c1, pw_c2 = compute_potential_ecoc_pos_weights(parquet_path, codebook, label_mode)

    # ------------------ Training loop ------------------
    best_score = math.inf

    for ep in range(1, epochs + 1):
        tr = train_one_epoch(
            model=model,
            model_name=model_name,
            
            loader=train_loader,
            optimizer=opt,
            device=device,
            
            lw=lw,
            
            pos_weight_c1=pw_c1,
            pos_weight_c2=pw_c2,
            label_mode=class_cfg.class_mode,
            
            uw=uw,
            debug_losses=debug_losses,
            
            regularize_hyperparams=regularize_hyperparams)

        va = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            lw=lw,
            pos_weight_c1=pw_c1,
            pos_weight_c2=pw_c2,
            label_mode=class_cfg.class_mode,
            codebook=codebook)

        print(f"[{ep:02d}] train: {tr}  |  val: {va}")

        # Combined score = rmse_km + classification penalty
        val_score = va["rmse_km"]
        if label_mode == "ecoc" and "c1_decoded_acc" in va and "c2_decoded_acc" in va:
            val_score += (1.0 - va["c1_decoded_acc"]) + (1.0 - va["c2_decoded_acc"])
        elif label_mode == "softmax" and "c1_top1" in va and "c2_top1" in va:
            val_score += (1.0 - va["c1_top1"]) + (1.0 - va["c2_top1"])

        if val_score < best_score:
            best_score = val_score
            ckpt = {
                "model": model.state_dict(),
                "uw": uw.state_dict() if uw is not None else 1.0,
                "config": {
                    "label_mode": class_cfg.class_mode,
                    "n_bits": getattr(class_cfg, "n_bits", None),
                    "n_classes_c1": getattr(class_cfg, "n_classes_c1", None),
                    "n_classes_c2": getattr(class_cfg, "n_classes_c2", None),
                    "layers": layer_counts,
                    "w0": w0,
                    "w_hidden": w_hidden,
                    "s": s,
                    "beta": beta,
                    "global_z": global_z,
                    "encoder_params": encoder_params,
                },
            }
            
            out_path = pathlib.Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)
            
            save_path = out_path / model_path
            
            torch.save(ckpt, save_path)
            print(f"  ↳ saved checkpoint: {save_path}")