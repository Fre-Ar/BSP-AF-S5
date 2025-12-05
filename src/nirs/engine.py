# src/nirs/engine.py

from __future__ import annotations

import torch

from geodata.ecoc.ecoc import (
    load_ecoc_codes,
    ecoc_prevalence_by_bit,
    pos_weight_from_prevalence,
)
from nirs.nns.nir import LabelMode
from nirs.create_nirs import build_model
from utils.utils import get_default_device

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
    device:  str | None  = None
):
    # resolve device
    device = device if device else get_default_device()
    
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
        
    first_param = next(model.parameters()).detach().flatten()

        
    return model, device, codebook, ckpt
