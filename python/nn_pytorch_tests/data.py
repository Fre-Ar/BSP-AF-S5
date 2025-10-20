# python/nn_pytorch_tests/data.py

import math, json, pathlib, random
from typing import Dict, Optional, Tuple, Literal
from dataclasses import dataclass
import torch
import torch.nn as nn, torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd, numpy as np
from tqdm import tqdm
import json
from loss import UncertaintyWeighting

# ===================== ECOC =====================

BIT_LENGTH = 32

def code_to_bits_np(code: int) -> np.ndarray:
    return np.array([(code >> b) & 1 for b in range(BIT_LENGTH)], dtype=np.uint8)

def load_ecoc_codes(path: str) -> dict[int, np.ndarray]:
    """Load ECOC map: class_id (int) -> np.array shape (n_bits,) with {0,1}."""
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): code_to_bits_np(int(v)) for k, v in raw.items()}


def _hamming_decode(bits_logits: torch.Tensor, codebook: Dict[int, np.ndarray]) -> torch.Tensor:
    """
    Argmin Hamming to nearest code.
    bits_logits: (B, n_bits) pre-sigmoid.
    Returns LongTensor of predicted class indices with same device.
    """
    with torch.no_grad():
        B, n_bits = bits_logits.shape
        # Convert codebook to tensor on the same device
        keys = sorted(codebook.keys())
        codes = torch.as_tensor(
            np.stack([codebook[k] for k in keys], axis=0), dtype=torch.float32, device=bits_logits.device
        )  # (C, n_bits) in {0,1}
        # turn logits into probabilities in (0,1) then to bits via threshold 0.5
        probs = torch.sigmoid(bits_logits)
        preds = (probs > 0.5).float()  # (B, n_bits)
        # Hamming distance = sum xor; since values are {0,1}, xor == |a-b|
        # Compute distance to each codebook row
        # (B, C, n_bits) -> (B, C)
        dists = (preds.unsqueeze(1) - codes.unsqueeze(0)).abs().sum(dim=-1)
        best = dists.argmin(dim=1)  # (B,)
        # Map back to true class IDs (keys)
        mapped = torch.as_tensor([keys[i] for i in best.tolist()], device=bits_logits.device, dtype=torch.long)
        return mapped


# ===================== DATA =====================

LabelMode = Literal["ecoc", "softmax"]

class BordersParquet(Dataset):
    """
    Parquet-backed dataset for training the spherical distance + country classification heads.

    Expects columns:
      lon, lat, x, y, z, dist_km, c1_id, c2_id, is_border, r_band
    (x,y,z) are the unit vector coords. dist_km is the geodesic distance to nearest border segment.
    
    ECOC targets:
      You may pass a codebook that maps class ID -> n_bits array (0/1). If label_mode="ecoc",
      it will output c1_bits and c2_bits FloatTensors.
        codebook: Dict[int, np.ndarray (n_bits,)]

    SOFTMAX targets:
      If label_mode="softmax", the dataset will output integer class indices (c1_idx, c2_idx).

    Splitting:
      - Deterministic shuffle with `seed`, then first chunk is 'train', tail is 'val'.
      - `split_frac=(train_frac, val_frac)` must sum to 1.

    Returns per item:
      {
        "xyz":      FloatTensor (3,),
        "dist":     FloatTensor (1,),            # in km
        "c1_bits":  FloatTensor (n_bits,),       # when label_mode="ecoc"
        "c2_bits":  FloatTensor (n_bits,),       # when label_mode="ecoc"
        "c1_idx":   LongTensor (),               # when label_mode="softmax"
        "c2_idx":   LongTensor (),               # when label_mode="softmax"
      }
    """
    def __init__(self,
                 parquet_path: str | pathlib.Path,
                 split: Literal["train", "val"] = "train",
                 split_frac: Tuple[float, float] = (0.9, 0.1),
                 seed: int = 1337,
                 codebook:  Optional[Dict[int, np.ndarray]] = None,
                 label_mode: LabelMode = "ecoc",
                 drop_cols: Tuple[str, ...] = ("lon", "lat", "is_border", "r_band")):
        super().__init__()
        assert abs(sum(split_frac) - 1.0) < 1e-6, "split_frac must sum to 1"
        assert split in ("train", "val")

        self.path = str(parquet_path)
        self.label_mode = label_mode
        # --- store codebook (keep as python dict) ---
        self.codebook = codebook

        # --- load dataframe (pyarrow backend can memory-map) ---
        df = pd.read_parquet(self.path)

        required = {"x", "y", "z", "dist_km", "c1_id", "c2_id"}
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
        c1_id = df["c1_id"].to_numpy(np.int32, copy=False)
        c2_id = df["c2_id"].to_numpy(np.int32, copy=False)
        
        # --- precompute tensors (fast __getitem__) ---
        # coordinates and regression target
        self.xyz = torch.from_numpy(self.df[["x","y","z"]].values).float()           # (N,3)
        self.dist = torch.from_numpy(self.df["dist_km"].values).float().unsqueeze(1) # (N,1)

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
            
        # Clean memory
        for c in drop_cols:
            if c in df:
                df = df.drop(columns=[c])
        
        # Class counts for softmax convenience
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
            "c1_idx":    self.c1_id[i],
            "c2_idx":    self.c2_id[i],   
        }
        if self.label_mode == "ecoc":
            item["c1_bits"] = self.c1_bits[i] # (32,)
            item["c2_bits"] = self.c2_bits[i] # (32,)
        return item

# ===================== TRAINING =====================

@dataclass
class LossWeights:
    w_dist: float = 1.0
    w_c1: float = 1.0
    w_c2: float = 1.0

def train_one_epoch(model: nn.Module,
                    loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device | str,
                    lw: LossWeights,
                    label_mode: LabelMode,
                    uw: UncertaintyWeighting | None = None
                    ) -> dict:
    """
    Trains for one epoch on batches containing:
      xyz -> model -> (pred_dist, c1_logits, c2_logits)
      dist (km), c1_bits (32), c2_bits (32)
      
    Returns a dict of running averages for logging.

    Loss:
      loss = w_dist * MSE(pred_dist, dist) +
             w_c1   * BCEWithLogits(c1_logits, c1_bits) +
             w_c2   * BCEWithLogits(c2_logits, c2_bits)

    Metrics returned:
      - rmse_km
      - c1_bit_acc, c2_bit_acc  (fraction of bits correct at 0.5 threshold)
      - loss
    """
    model.train()
    totals = {
        "loss": 0.0,
        "mse_sum": 0.0,
        "n": 0,
        "c1_loss": 0.0,
        "c2_loss": 0.0,
    }
    
    mse = nn.MSELoss(reduction="mean")
    bce = nn.BCEWithLogitsLoss(reduction="mean")
    ce = nn.CrossEntropyLoss(reduction="mean")
    
    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        xyz       = batch["xyz"].to(device)           # (B,3)
        dist      = batch["dist"].to(device)          # (B,1)

        optimizer.zero_grad(set_to_none=True)

        pred_dist, c1_logits, c2_logits = model(xyz)  # (B,1), (B,32), (B,32)
        # Distance regression
        loss_dist = mse(pred_dist, dist)
        
        # Classification
        if label_mode == "ecoc":
            c1_bits = batch["c1_bits"].to(device)
            c2_bits = batch["c2_bits"].to(device)
            loss_c1 = bce(c1_logits, c1_bits)
            loss_c2 = bce(c2_logits, c2_bits)
        else:
            c1_idx = batch["c1_idx"].to(device)
            c2_idx = batch["c2_idx"].to(device)
            loss_c1 = ce(c1_logits, c1_idx)
            loss_c2 = ce(c2_logits, c2_idx)
        
       
        if uw:
            loss = uw([lw.w_dist * loss_dist, lw.w_c1 * loss_c1, lw.w_c2 * loss_c2])
        else:
            loss = lw.w_dist * loss_dist + lw.w_c1 * loss_c1 + lw.w_c2 * loss_c2
        loss.backward()
        
        optimizer.step()

        with torch.no_grad():
            B = xyz.size(0)
            totals["loss"] += float(loss.detach()) * B
            totals["mse_sum"] += F.mse_loss(pred_dist.detach(), dist, reduction="sum").float()
            totals["c1_loss"] += float(loss_c1.detach()) * B
            totals["c2_loss"] += float(loss_c2.detach()) * B
            totals["n"] += B
            
            # bit accuracies
            #c1_pred_bits = (torch.sigmoid(c1_logits) >= 0.5).float()
            #c2_pred_bits = (torch.sigmoid(c2_logits) >= 0.5).float()
            #running["bits1"] += (c1_pred_bits.eq(c1_bits)).float().sum().item()
            #running["bits2"] += (c2_pred_bits.eq(c2_bits)).float().sum().item()


            pbar.set_postfix({
                "rmse(km)": (totals["mse_sum"]/totals["n"])**0.5,
                "c1_loss/n": totals["c1_loss"] / max(1, totals["n"]),
                "c2_loss/n": totals["c2_loss"] / max(1, totals["n"]),
            })
    
    n = max(1, totals["n"])
    return {
        "loss":      totals["loss"] / n,
        "rmse_km":   math.sqrt(totals["mse_sum"] / n),
        "c1_loss":   totals["c1_loss"] / n,
        "c2_loss":   totals["c2_loss"] / n,
    }

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device | str,
             lw: LossWeights,
             label_mode: LabelMode,
             codebook: Optional[Dict[int, np.ndarray]] = None,
             uw: UncertaintyWeighting | None = None):
    """
    Validation loop (no grad). Expects batches with: xyz, dist, c1_bits, c2_bits.
    Model must return: (pred_dist, c1_logits, c2_logits).

    For ECOC we report bit-accuracy and (if codebook) decoded accuracy.
    For softmax we report top-1 accuracy.
    
    Returns:
      {
        "loss": average total loss over samples,
        "rmse_km": sqrt(sum(MSE)/N),
        "c1_bit_acc": fraction of correct bits for c1 head,
        "c2_bit_acc": fraction of correct bits for c2 head,
      }
    """
    model.eval()
    totals = {
        "mse_sum": 0.0,
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
            dist    = batch["dist"].to(device)

            pred_dist, c1_logits, c2_logits = model(xyz)
            
            B = xyz.shape[0]
            # Distance Regression
            totals["mse_sum"] += lw.w_dist * F.mse_loss(pred_dist, dist, reduction="sum").float()
            totals["n"] += B

            # Classification
            if label_mode == "ecoc":
                c1_bits = batch["c1_bits"].to(device)
                c2_bits = batch["c2_bits"].to(device)
                # bit accuracy
                c1_pred_bits = (torch.sigmoid(c1_logits) > 0.5).float()
                c2_pred_bits = (torch.sigmoid(c2_logits) > 0.5).float()

                totals["c1_bits_correct"] += float((c1_pred_bits.eq(c1_bits)).float().sum().item())
                totals["c2_bits_correct"] += float((c2_pred_bits.eq(c2_bits)).float().sum().item())
                totals["n_bits_total"] += int(c1_bits.numel() + c2_bits.numel())

                # decoded top-1 (requires codebook)
                if codebook is not None:
                    c1_idx_true = batch.get("c1_idx")
                    c2_idx_true = batch.get("c2_idx")
                    # If dataset didn't include idx for ecoc mode, create from codebook mapping order.
                    if c1_idx_true is None or c2_idx_true is None:
                        # we skip decoded accuracy if not available.
                        pass
                    else:
                        c1_idx_true = c1_idx_true.to(device)
                        c2_idx_true = c2_idx_true.to(device)
                        c1_idx_pred = _hamming_decode(c1_logits, codebook)
                        c2_idx_pred = _hamming_decode(c2_logits, codebook)
                        totals["c1_decoded_top1"] += (c1_idx_pred == c1_idx_true).float().sum().item()
                        totals["c2_decoded_top1"] += (c2_idx_pred == c2_idx_true).float().sum().item()

            else:  # softmax
                c1_idx = batch["c1_idx"].to(device)
                c2_idx = batch["c2_idx"].to(device)
                c1_top1 = c1_logits.argmax(dim=1).eq(c1_idx).float().sum().item()
                c2_top1 = c2_logits.argmax(dim=1).eq(c2_idx).float().sum().item()
                totals["c1_top1"] += float(c1_top1)
                totals["c2_top1"] += float(c2_top1)

    n = max(1, totals["n"])
    out = {
        "rmse_km": math.sqrt(totals["mse_sum"] / n),
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

