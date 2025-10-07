import math, json, pathlib, random
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pandas as pd
from tqdm import tqdm
from siren import MultiHeadSIREN

# ===================== DATA =====================

@dataclass
class LabelMaps:
    c1_to_idx: dict
    c2_to_idx: dict
    idx_to_c1: list
    idx_to_c2: list

class BordersParquet(Dataset):
    """
    Loads a Parquet table with columns:
      lon, lat, x, y, z, dist_km, c1_id, c2_id, is_border, r_band
    Builds contiguous label ids for c1_id and c2_id.
    """
    def __init__(self, parquet_path, split="train", split_frac=(0.95, 0.05),
                 cache_dir=None, seed=1337, use_columns=None):
        super().__init__()
        self.path = str(parquet_path)
        self.use_columns = use_columns or ["x","y","z","dist_km","c1_id","c2_id","is_border","r_band"]

        # Load once (pandas uses pyarrow backend, memory-maps when possible)
        df = pd.read_parquet(self.path, columns=self.use_columns)

        # Shuffle + split
        rng = random.Random(seed)
        idx = list(range(len(df)))
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * split_frac[0])
        ids = idx[:n_train] if split=="train" else idx[n_train:]
        self.df = df.iloc[ids].reset_index(drop=True)

        # Build contiguous maps (persistable)
        c1_vals = pd.Index(df["c1_id"].unique()).sort_values()
        c2_vals = pd.Index(df["c2_id"].unique()).sort_values()
        self.c1_to_idx = {int(v): i for i,v in enumerate(c1_vals)}
        self.c2_to_idx = {int(v): i for i,v in enumerate(c2_vals)}
        self.idx_to_c1 = [int(v) for v in c1_vals]
        self.idx_to_c2 = [int(v) for v in c2_vals]

        # Precompute tensors (speeds up training)
        self.xyz = torch.from_numpy(self.df[["x","y","z"]].values).float()
        self.dist = torch.from_numpy(self.df["dist_km"].values).float().unsqueeze(1)
        self.c1 = torch.tensor([self.c1_to_idx[int(v)] for v in self.df["c1_id"].values], dtype=torch.long)
        self.c2 = torch.tensor([self.c2_to_idx[int(v)] for v in self.df["c2_id"].values], dtype=torch.long)

        # Optional save of label maps
        if cache_dir:
            cache_dir = pathlib.Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
            with open(cache_dir/"label_maps.json","w") as f:
                json.dump({
                    "c1_to_idx": self.c1_to_idx,
                    "c2_to_idx": self.c2_to_idx,
                    "idx_to_c1": self.idx_to_c1,
                    "idx_to_c2": self.idx_to_c2
                }, f)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        return {
            "xyz": self.xyz[i],
            "dist": self.dist[i],
            "c1": self.c1[i],
            "c2": self.c2[i]
        }
        
# ===================== TRAINING =====================

@dataclass
class LossWeights:
    w_dist: float = 1.0
    w_c1: float = 1.0
    w_c2: float = 1.0

def train_one_epoch(model, loader, opt, device, lw: LossWeights,
                    grad_clip=1.0, max_dist_km=None):
    model.train()
    ce = nn.CrossEntropyLoss()
    running = {"loss":0.0,"mse":0.0,"c1_acc":0.0,"c2_acc":0.0,"n":0}

    for batch in tqdm(loader, leave=False):
        xyz = batch["xyz"].to(device)
        dist = batch["dist"].to(device)
        c1 = batch["c1"].to(device)
        c2 = batch["c2"].to(device)

        pred_dist, c1_logits, c2_logits = model(xyz)

        # Optionally clamp target distances (for outlier robustness)
        if max_dist_km is not None:
            dist = dist.clamp_max(max_dist_km)

        mse = F.mse_loss(pred_dist, dist)
        loss = lw.w_dist * mse

        if lw.w_c1 > 0:
            loss += lw.w_c1 * ce(c1_logits, c1)
        if lw.w_c2 > 0:
            loss += lw.w_c2 * ce(c2_logits, c2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        with torch.no_grad():
            running["loss"] += loss.item() * xyz.size(0)
            running["mse"]  += mse.item() * xyz.size(0)
            running["c1_acc"] += (c1_logits.argmax(1)==c1).float().sum().item()
            running["c2_acc"] += (c2_logits.argmax(1)==c2).float().sum().item()
            running["n"] += xyz.size(0)

    n = max(1, running["n"])
    return {
        "loss": running["loss"]/n,
        "rmse_km": math.sqrt(running["mse"]/n),
        "c1_acc": running["c1_acc"]/n,
        "c2_acc": running["c2_acc"]/n
    }

@torch.no_grad()
def evaluate(model, loader, device, lw: LossWeights, max_dist_km=None):
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    s = {"loss":0.0,"mse":0.0,"c1_acc":0.0,"c2_acc":0.0,"n":0}

    for batch in loader:
        xyz = batch["xyz"].to(device)
        dist = batch["dist"].to(device)
        c1 = batch["c1"].to(device)
        c2 = batch["c2"].to(device)

        pred_dist, c1_logits, c2_logits = model(xyz)
        if max_dist_km is not None:
            dist = dist.clamp_max(max_dist_km)

        mse = F.mse_loss(pred_dist, dist, reduction="sum")
        loss = lw.w_dist*mse
        if lw.w_c1>0: loss += lw.w_c1 * ce(c1_logits, c1)
        if lw.w_c2>0: loss += lw.w_c2 * ce(c2_logits, c2)

        s["loss"] += loss.item()
        s["mse"]  += mse.item()
        s["c1_acc"] += (c1_logits.argmax(1)==c1).float().sum().item()
        s["c2_acc"] += (c2_logits.argmax(1)==c2).float().sum().item()
        s["n"] += xyz.size(0)

    n = max(1, s["n"])
    return {
        "loss": s["loss"]/n,
        "rmse_km": math.sqrt(s["mse"]/n),
        "c1_acc": s["c1_acc"]/n,
        "c2_acc": s["c2_acc"]/n
    }

# ===================== MAIN =====================

def main(parquet_path, batch_size=8192, epochs=10,
         hidden=256, depth=5, w0_first=30.0, w0_hidden=1.0,
         lr=9e-4, loss_weights=(1.0,1.0,1.0),
         max_dist_km=None, device=None):

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_ds = BordersParquet(parquet_path, split="train", split_frac=(0.95,0.05))
    val_ds   = BordersParquet(parquet_path, split="val",   split_frac=(0.95,0.05))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)

    # Model
    model = MultiHeadSIREN(in_dim=3, hidden=hidden, depth=depth,
                           w0_first=w0_first, w0_hidden=w0_hidden,
                           num_c1=len(train_ds.idx_to_c1), num_c2=len(train_ds.idx_to_c2)).to(device)

    # Optim
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    lw = LossWeights(*loss_weights)
    print(f"Classes: c1={len(train_ds.idx_to_c1)}, c2={len(train_ds.idx_to_c2)}")
    print(f"Training on device: {device}")

    best = None
    for ep in range(1, epochs+1):
        tr = train_one_epoch(model, train_loader, opt, device, lw, max_dist_km=max_dist_km)
        va = evaluate(model, val_loader, device, lw, max_dist_km=max_dist_km)
        line = (f"[{ep:03d}] "
                f"train rmse={tr['rmse_km']:.3f}km c1={tr['c1_acc']:.3f} c2={tr['c2_acc']:.3f} | "
                f"val rmse={va['rmse_km']:.3f}km c1={va['c1_acc']:.3f} c2={va['c2_acc']:.3f}")
        print(line)
        # Track best by val RMSE + (1-acc) penalties
        score = va["rmse_km"] + (1.0 - va["c1_acc"]) + (1.0 - va["c2_acc"])
        if best is None or score < best[0]:
            best = (score, {k:float(v) for k,v in va.items()})
            torch.save({"model":model.state_dict(),
                        "label_maps":{
                            "idx_to_c1": train_ds.idx_to_c1,
                            "idx_to_c2": train_ds.idx_to_c2
                        }}, "siren_multitask_best.pt")
            print("  ↳ saved checkpoint: siren_multitask_best.pt")

if __name__ == "__main__":
    PATH = "python/geodata/parquet/dataset_all.parquet"
    main(PATH,
         epochs=20,
         batch_size=8192,
         hidden=256,
         depth=5,
         w0_first=30.0,
         w0_hidden=1.0,
         lr=9e-4,
         loss_weights=(1.0,1.0,1.0),
         max_dist_km=None)