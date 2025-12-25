# src/nirs/engine.py

from __future__ import annotations

from typing import Union
import torch
import numpy as np

from geodata.ecoc.ecoc import (
    load_ecoc_codes,
    ecoc_prevalence_by_bit,
    pos_weight_from_prevalence,
    ecoc_decode,
    _prepare_codebook_tensor
)
from nirs.nns.nir import LabelMode
from nirs.create_nirs import build_model
from utils.utils import get_default_device
from .inference import InferenceConfig, Prediction

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
        # We reuse the engine logic to ensure consistency with training
        self.model, self.device, self.codebook, self.ckpt = load_model_and_codebook(
            checkpoint_path=checkpoint_path,
            model_name=cfg.model_name,
            layer_counts=cfg.layer_counts,
            w0=cfg.w0,
            w_hidden=cfg.w_hidden,
            s_param=cfg.s,
            beta=cfg.beta,
            global_z=cfg.global_z,
            label_mode=cfg.label_mode,
            codes_path=cfg.codes_path,
            device=self.device
        )
        self.model.eval()
        
        # 2. Prepare Codebook Tensors (for ECOC decoding)
        if self.cfg.label_mode == "ecoc":
            self.class_ids, self.codes_mat = _prepare_codebook_tensor(
                self.codebook, self.device, torch.float32
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

    def _forward_model(self, x: torch.Tensor):
        """Unified forward pass handling model-specific signatures (e.g. INCODE)."""
        if self.cfg.model_name.lower() == "incode":
            # INCODE returns (dist, c1, c2, reg_params)
            out = self.model(x, self.cfg.regularize_hyperparams)
            return out[0], out[1], out[2]
        else:
            # Standard return (dist, c1, c2)
            return self.model(x)

    @torch.no_grad()
    def predict(self, xyz: Union[np.ndarray, torch.Tensor], tau: float = 1.0) -> Prediction:
        """
        Runs inference on a batch of points.
        
        Args:
            xyz: (N, 3) array or tensor of unit vectors.
            tau: Temperature for ECOC decoding (default 1.0).
            
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
            c1_ids = c1_logits.argmax(dim=1)
            c2_ids = c2_logits.argmax(dim=1)
            
        return Prediction(
            dist_km=dist_km.cpu().numpy(),
            c1_ids=c1_ids.long().cpu().numpy(),
            c2_ids=c2_ids.long().cpu().numpy()
        )