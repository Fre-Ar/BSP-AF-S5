# src/nirs/engine.py

from __future__ import annotations
from typing import Tuple

import torch
from torch.utils.data import DataLoader

from geodata.ecoc.ecoc import (
    load_ecoc_codes,
    ecoc_prevalence_by_bit,
    pos_weight_from_prevalence,
)
from nirs.nns.nir import ClassHeadConfig, LabelMode
from nirs.training import BordersParquet
from nirs.create_nirs import build_model
from utils.utils import get_default_device

def make_dataloaders(
    parquet_path: str,
    *,
    label_mode: LabelMode,
    codes_path: str | None,
    split: Tuple[float, float] = (0.9, 0.1),
    batch_size: int = 8192,
    codebook: dict | None = None,
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
        label_mode=label_mode)
    
    val_ds = BordersParquet(
        parquet_path,
        split="val",
        split_frac=split,
        codebook=codebook,
        label_mode=label_mode)
    
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

    # create the dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader, class_cfg, codebook


def compute_potential_ecoc_pos_weights(
    parquet_path: str,
    codebook: dict,
    label_mode: LabelMode):
    """
    Computes ECOC per-bit pos_weight for c1/c2 over a full parquet.
    Returns None, None if label_mode == 'softmax'.

    Returns
    -------
    pw_c1, pw_c2 : torch.Tensor
        Suitable for BCEWithLogitsLoss(pos_weight=...).
    """
    if codebook is None:
        return None, None
    
    if label_mode == "ecoc":
        # BCE bit prevalence correction
        ones_c1, totals_c1, p1_c1 = ecoc_prevalence_by_bit(parquet_path, codebook, id_col="c1_id")
        ones_c2, totals_c2, p1_c2 = ecoc_prevalence_by_bit(parquet_path, codebook, id_col="c2_id")

        pw_c1 = pos_weight_from_prevalence(p1_c1)
        pw_c2 = pos_weight_from_prevalence(p1_c2)
    else:
        pw_c1 = pw_c2 = None

    return pw_c1, pw_c2

def load_model_and_codebook(
    checkpoint_path: str,
    model_name: str,
    layer_counts,
    w0: float,
    w_hidden: float,
    s_param: float,
    beta: float,
    global_z: bool,
    label_mode: LabelMode = "ecoc",
    codes_path: str | None = None,
    device:  str | None  = None,
):
    # resolve device
    device = device if device else device = get_default_device()
    
    # ---- load checkpoint & config ----
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    cfg = ckpt.get("config", {})

    if label_mode not in {"ecoc", "softmax"}:
        raise ValueError(f"label_mode must be 'auto'|'ecoc'|'softmax', got {label_mode}")

    # ---- build & load model ----
    model, _ = build_model(
        model_name,
        layer_counts,
        label_mode,
        (w0, w_hidden, s_param, beta, global_z),
    )
    model.to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    codebook = None
    if label_mode == "ecoc":
        if codes_path is None:
            raise ValueError("ECOC mode requires codes_path to the ECOC JSON codebook.")
        codebook = load_ecoc_codes(codes_path)

    return model, device, codebook, ckpt
