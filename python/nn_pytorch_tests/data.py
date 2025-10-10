import math, json, pathlib, random
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn as nn, torch.optim as opt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd, numpy as np
from tqdm import tqdm
import json

# ===================== ECOC =====================

BIT_LENGTH = 32

def code_to_bits_np(code: int) -> np.ndarray:
    return np.array([(code >> b) & 1 for b in range(BIT_LENGTH)], dtype=np.uint8)

def load_ecoc_codes(path: str) -> dict[int, np.ndarray]:
    """Load id->32-bit code from JSON, converting keys to int."""
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): code_to_bits_np(int(v)) for k, v in raw.items()}




# ===================== DATA =====================


class BordersParquet(Dataset):
    """
    Parquet-backed dataset for multi-head ECOC training.

    Expects columns:
      lon, lat, x, y, z, dist_km, c1_id, c2_id, is_border, r_band

    ECOC targets:
      You must pass a codebook that maps class ID -> 32-bit array (0/1).
        codebook: Dict[int, np.ndarray (32,)]

    Splitting:
      - Deterministic shuffle with `seed`, then first chunk is 'train', tail is 'val'.
      - `split_frac=(train_frac, val_frac)` must sum to 1.

    Returns per item:
      {
        "xyz":      FloatTensor (3,),
        "dist":     FloatTensor (1,),            # in km
        "c1_bits":  FloatTensor (32,),           # {0.,1.}
        "c2_bits":  FloatTensor (32,),           # {0.,1.}
      }
    """
    def __init__(self,
                 parquet_path: str | pathlib.Path,
                 codebook: Dict[int, np.ndarray],
                 split: str = "train",
                 split_frac: Tuple[float, float] = (0.95, 0.05),
                 seed: int = 1337,
                 use_columns: Optional[list] = None,
                 cache_dir: Optional[str | pathlib.Path] = None):
        super().__init__()
        assert abs(sum(split_frac) - 1.0) < 1e-6, "split_frac must sum to 1"
        assert split in ("train", "val")

        self.path = str(parquet_path)
        self.use_columns = use_columns or [
            "x","y","z","dist_km","c1_id","c2_id","is_border","r_band"
        ]

        # --- load dataframe (pyarrow backend can memory-map) ---
        df = pd.read_parquet(self.path, columns=self.use_columns)

        # --- deterministic shuffle & split ---
        rng = random.Random(seed)
        idx = list(range(len(df)))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * split_frac[0])
        take = idx[:n_train] if split == "train" else idx[n_train:]
        self.df = df.iloc[take].reset_index(drop=True)

        # --- sanity: all IDs present in codebooks ---
        all_ids_needed = set(pd.unique(pd.concat([df["c1_id"], df["c2_id"]], ignore_index=True)).tolist())
        # set difference
        missing = all_ids_needed - set(codebook.keys())
        if missing:
            raise ValueError(
                f"Codebook is missing {len(missing)} id(s) present in the dataset, "
                f"e.g. {sorted(list(missing))[:5]}"   
            )

        # --- store codebook (keep as python dict) ---
        self.codebook = codebook

        # --- precompute tensors (fast __getitem__) ---
        # coordinates and regression target
        self.xyz = torch.from_numpy(self.df[["x","y","z"]].values).float()           # (N,3)
        self.dist = torch.from_numpy(self.df["dist_km"].values).float().unsqueeze(1) # (N,1)


        # ECOC bits (float32 in {0.,1.})
        def _stack_bits(ids, codebook):
            bits = [codebook[int(v)] for v in ids]
            arr = np.stack(bits, 0).astype(np.float32)  # (N,32)
            if arr.shape[1] != BIT_LENGTH:
                raise ValueError(f"Expected {BIT_LENGTH}-bit codes, got shape {arr.shape}")
            return torch.from_numpy(arr)

        self.c1_bits = _stack_bits(self.df["c1_id"].values, self.codebook)  # (N,32)
        self.c2_bits = _stack_bits(self.df["c2_id"].values, self.codebook)  # (N,32)

        # --- optional: persist a small manifest for reproducibility ---
        if cache_dir:
            cache = pathlib.Path(cache_dir)
            cache.mkdir(parents=True, exist_ok=True)
            manifest = {
                "num_rows": len(self.df),
                "path": self.path,
                "columns": self.use_columns,
                "split": split,
                "split_frac": split_frac,
                "seed": seed,
                "ids_seen": sorted([int(v) for v in all_ids_needed]),
            }
            with open(cache/"dataset_manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        return {
            "xyz":       self.xyz[i],        # (3,)
            "dist":      self.dist[i],       # (1,)
            "c1_bits":   self.c1_bits[i],    # (32,)
            "c2_bits":   self.c2_bits[i],    # (32,)
        }

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
                    grad_clip: Optional[float] = 1.0,
                    max_dist_km: Optional[float] = None,
                    pos_weight_c1: Optional[torch.Tensor] = None,
                    pos_weight_c2: Optional[torch.Tensor] = None
                    ) -> dict:
    """
    Trains for one epoch on batches containing:
      xyz -> model -> (pred_dist, c1_logits, c2_logits)
      dist (km), c1_bits (32), c2_bits (32)

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
    
    if pos_weight_c1 is not None:
        pos_weight_c1 = pos_weight_c1.to(device)  # shape (32,)
    if pos_weight_c2 is not None:
        pos_weight_c2 = pos_weight_c2.to(device)

    bce_c1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight_c1)
    bce_c2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight_c2)

    running = {"loss": 0.0, "mse_sum": 0.0, "bits1": 0.0, "bits2": 0.0, "n": 0, "bitsN": 0}

    pbar = tqdm(loader, leave=False)
    for batch in pbar:
        xyz       = batch["xyz"].to(device)           # (B,3)
        dist      = batch["dist"].to(device)          # (B,1)
        c1_bits   = batch["c1_bits"].to(device)       # (B,32)
        c2_bits   = batch["c2_bits"].to(device)       # (B,32)

        if max_dist_km is not None:
            dist = dist.clamp_max(max_dist_km)

        optimizer.zero_grad(set_to_none=True)


        pred_dist, c1_logits, c2_logits = model(xyz)  # (B,1), (B,32), (B,32)

        mse = F.mse_loss(pred_dist, dist)
        loss = lw.w_dist * mse
        if lw.w_c1 > 0:
            loss += lw.w_c1 * bce_c1(c1_logits, c1_bits)
        if lw.w_c2 > 0:
            loss += lw.w_c2 * bce_c2(c2_logits, c2_bits)

        loss.backward()
        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        with torch.no_grad():
            B = xyz.size(0)
            running["loss"]   += loss.item() * B
            running["mse_sum"]+= F.mse_loss(pred_dist, dist, reduction="sum").item()
            # bit accuracies
            c1_pred_bits = (torch.sigmoid(c1_logits) >= 0.5).float()
            c2_pred_bits = (torch.sigmoid(c2_logits) >= 0.5).float()
            running["bits1"] += (c1_pred_bits.eq(c1_bits)).float().sum().item()
            running["bits2"] += (c2_pred_bits.eq(c2_bits)).float().sum().item()
            running["bitsN"] += 2 * B * c1_bits.size(1)  # total number of compared bits across both heads
            running["n"]     += B

            pbar.set_postfix({
                "rmse(km)": (running["mse_sum"]/running["n"])**0.5,
                "b1": running["bits1"] / max(1, running["bitsN"]/2),
                "b2": running["bits2"] / max(1, running["bitsN"]/2),
            })

    n = max(1, running["n"])
    bit_den = max(1, running["bitsN"]//2)  # per-head denominator (B*32 summed across batches)
    return {
        "loss":     running["loss"] / n,
        "rmse_km":  (running["mse_sum"] / n) ** 0.5,
        "c1_bit_acc": running["bits1"] / bit_den,
        "c2_bit_acc": running["bits2"] / bit_den,
    }

@torch.no_grad()
def evaluate(model: nn.Module,
             loader: DataLoader,
             device: torch.device | str,
             lw: LossWeights,
             max_dist_km: Optional[float] = None,
             pos_weight_c1: Optional[torch.Tensor] = None,
             pos_weight_c2: Optional[torch.Tensor] = None) -> dict:
    """
    Validation loop (no grad). Expects batches with: xyz, dist, c1_bits, c2_bits.
    Model must return: (pred_dist, c1_logits, c2_logits).

    Returns:
      {
        "loss": average total loss over samples,
        "rmse_km": sqrt(sum(MSE)/N),
        "c1_bit_acc": fraction of correct bits for c1 head,
        "c2_bit_acc": fraction of correct bits for c2 head,
      }
    """
    model.eval()

    if pos_weight_c1 is not None:
        pos_weight_c1 = pos_weight_c1.to(device)
    if pos_weight_c2 is not None:
        pos_weight_c2 = pos_weight_c2.to(device)

    # Use sum-reduction to aggregate exactly over the whole epoch
    bce_c1 = nn.BCEWithLogitsLoss(pos_weight=pos_weight_c1, reduction="sum")
    bce_c2 = nn.BCEWithLogitsLoss(pos_weight=pos_weight_c2, reduction="sum")

    totals = {
        "loss_sum": 0.0,
        "mse_sum":  0.0,
        "bits1":    0.0,
        "bits2":    0.0,
        "bitsN":    0,    # total count of compared bits across both heads
        "n":        0,    # total samples
    }

    for batch in loader:
        xyz     = batch["xyz"].to(device)
        dist    = batch["dist"].to(device)
        c1_bits = batch["c1_bits"].to(device)
        c2_bits = batch["c2_bits"].to(device)

        if max_dist_km is not None:
            dist = dist.clamp_max(max_dist_km)

        pred_dist, c1_logits, c2_logits = model(xyz)

        # sum-reduction MSE
        mse_sum = F.mse_loss(pred_dist, dist, reduction="sum")
        loss_sum = lw.w_dist * mse_sum
        if lw.w_c1 > 0:
            loss_sum = loss_sum + lw.w_c1 * bce_c1(c1_logits, c1_bits)
        if lw.w_c2 > 0:
            loss_sum = loss_sum + lw.w_c2 * bce_c2(c2_logits, c2_bits)

        # bit accuracies
        c1_pred = (torch.sigmoid(c1_logits) >= 0.5).float()
        c2_pred = (torch.sigmoid(c2_logits) >= 0.5).float()
        bits1 = (c1_pred.eq(c1_bits)).float().sum().item()
        bits2 = (c2_pred.eq(c2_bits)).float().sum().item()

        B = xyz.size(0)
        totals["loss_sum"] += loss_sum.item()
        totals["mse_sum"]  += mse_sum.item()
        totals["bits1"]    += bits1
        totals["bits2"]    += bits2
        totals["bitsN"]    += 2 * B * c1_bits.size(1)
        totals["n"]        += B

    n = max(1, totals["n"])
    per_head_bits = max(1, totals["bitsN"] // 2)

    return {
        "loss":        totals["loss_sum"] / n,
        "rmse_km":     (totals["mse_sum"] / n) ** 0.5,
        "c1_bit_acc":  totals["bits1"] / per_head_bits,
        "c2_bit_acc":  totals["bits2"] / per_head_bits,
    }

