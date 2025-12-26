# src/nirs/data.py

from typing import Dict, Optional, Tuple, Literal
import pathlib
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from nirs.nns.nir import ClassHeadConfig, LabelMode
from geodata.ecoc.ecoc import load_ecoc_codes
from utils.utils_geo import SEED

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
            "c1_id":      LongTensor (),           # raw class id
            "c2_id":      LongTensor (),
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
        
        drop_cols: Tuple[str, ...] = ("lon", "lat", "is_border"),
        device=None
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
        
        if device is not None:
            self.xyz = self.xyz.to(device)
            self.dist = self.dist.to(device)
            self.log1p_dist = self.log1p_dist.to(device)
            self.r_band = self.r_band.to(device)
            self.c1_id = self.c1_id.to(device)
            self.c2_id = self.c2_id.to(device)
            if self.label_mode == "ecoc":
                self.c1_bits = self.c1_bits.to(device)
                self.c2_bits = self.c2_bits.to(device)

    def __len__(self) -> int:
        return self.xyz.shape[0]

    def __getitem__(self, i: int) -> dict:
        item = {
            "xyz":       self.xyz[i],        # (3,)
            "dist":      self.dist[i],       # (1,)
            "log1p_dist":self.log1p_dist[i],       # (1,)
            "r_band":    self.r_band[i],
            "c1_id":     self.c1_id[i],
            "c2_id":     self.c2_id[i],
        }
        if self.label_mode == "ecoc":
            item["c1_bits"] = self.c1_bits[i]  # (n_bits,)
            item["c2_bits"] = self.c2_bits[i]  # (n_bits,)
        return item

class FastTensorDataLoader:
    """
    Custom iterator that slices batches directly from GPU tensors.
    Replaces standard DataLoader when dataset is already on device.
    """
    def __init__(self, dataset: BordersParquet, batch_size: int, shuffle: bool = True):
        self.ds = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.size = len(dataset)
        
    def __iter__(self):
        # Generate indices on the same device as data (avoid cpu<->gpu transfer)
        if self.shuffle:
            indices = torch.randperm(self.size, device=self.ds.xyz.device)
        else:
            indices = torch.arange(self.size, device=self.ds.xyz.device)

        # Iterate via slicing
        for start in range(0, self.size, self.batch_size):
            end = start + self.batch_size
            idx = indices[start:end]
            
            # Slice Big Tensors -> Batch Tensors (1 Kernel call)
            batch = {
                "xyz": self.ds.xyz[idx],
                "dist": self.ds.dist[idx],
                "log1p_dist": self.ds.log1p_dist[idx],
                "r_band": self.ds.r_band[idx],
                "c1_id": self.ds.c1_id[idx],
                "c2_id": self.ds.c2_id[idx],
            }
            
            if self.ds.label_mode == "ecoc":
                batch["c1_bits"] = self.ds.c1_bits[idx]
                batch["c2_bits"] = self.ds.c2_bits[idx]
                
            yield batch

    def __len__(self):
        return (self.size + self.batch_size - 1) // self.batch_size

def make_dataloaders(
    parquet_path: str,
    *,
    label_mode: LabelMode,
    codes_path: str | None,
    split: Tuple[float, float] = (0.9, 0.1),
    batch_size: int = 8192,
    codebook: dict | None = None,
    device=None
):
    """
    Shared dataset + dataloader construction for training & eval.

    - In ECOC mode, loads ECOC codebook (if not provided) and builds
      BordersParquet with bit targets.
    - In softmax mode, builds BordersParquet with integer targets.
    """
    if label_mode == "ecoc":
        if codebook is None:
            assert codes_path is not None, "codes_path is required for ECOC mode."
            codebook = load_ecoc_codes(codes_path)
    
    # create the parquet wrapper objects
    train_ds = BordersParquet(
        parquet_path,
        split="train",
        split_frac=split,
        codebook=codebook,
        label_mode=label_mode,
        device=device)
    
    val_ds = BordersParquet(
        parquet_path,
        split="val",
        split_frac=split,
        codebook=codebook,
        label_mode=label_mode,
        device=device)
    
    if label_mode == "ecoc":
        bits = len(next(iter(codebook.values())))
        class_cfg = ClassHeadConfig(
            class_mode=label_mode,
            n_bits=bits)
    else:
        class_cfg = ClassHeadConfig(
            class_mode=label_mode,
            n_classes_c1=train_ds.num_classes_c1,
            n_classes_c2=train_ds.num_classes_c2)
    
    # 3. Choose Loader Strategy
    if device is not None:
        # GPU-Resident Mode: Use Fast Loader
        train_loader = FastTensorDataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = FastTensorDataLoader(val_ds, batch_size=batch_size, shuffle=False)
    else:
        # CPU Mode: Use Standard Loader
        from utils.utils import get_default_device
        def_device = get_default_device()
        use_pin_memory = str(def_device) == "cuda"
        
        # create the dataloaders
        workers = 4
        keep_workers = True
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers= workers,
            pin_memory=use_pin_memory,
            persistent_workers=keep_workers
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers= workers,
            pin_memory=use_pin_memory,
            persistent_workers=keep_workers
        )
    
    
    return train_loader, val_loader, class_cfg, codebook

