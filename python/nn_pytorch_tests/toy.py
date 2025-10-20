# python/nn_pytorch_tests/toy.py

from nn_siren import SIRENLayer
from nn_relu import ReLULayer
from fourier_features import PositionalEncoding
from nir import NIRLayer, NIRTrunk, MultiHeadNIR, ClassHeadConfig
import torch, torch.nn as nn, torch.nn.functional as F
import math 
from torch.utils.data import DataLoader
from data import BordersParquet, LossWeights, train_one_epoch, evaluate, load_ecoc_codes, BIT_LENGTH    
import pathlib
from loss import UncertaintyWeighting

# ===================== MAIN =====================

LARGEST_DIST = 4000

def main(parquet_path, codes_path="python/geodata/countries.ecoc.json",
         batch_size=8192, epochs=10,
         layer_counts=(256,)*5, w0=30.0, w_hidden=1.0,
         lr=9e-4, loss_weights=(1.0,1.0,1.0),
         n_bits=BIT_LENGTH,
         label_mode: str = "ecoc",  # 'ecoc' or 'softmax'
         country_n: int = 289,
         device: str | None = None):

    print(f"Training on device: {device}")
    
    # Model
    depth = len(layer_counts)
    model = MultiHeadNIR(SIRENLayer,
                         in_dim=3,
                         layer_counts=layer_counts,
                         params=((w0,),)+((w_hidden,),)*(depth-1),
                         class_cfg = ClassHeadConfig(class_mode=label_mode,
                                                     n_bits=n_bits,
                                                     n_classes_c1=country_n,
                                                     n_classes_c2=country_n)
                         ).to(device)
    '''model = MultiHeadNIR(ReLULayer,
                        in_dim=3,
                        layer_counts=layer_counts,
                        params=((),)+((),)*(depth-1),
                        class_cfg = ClassHeadConfig(class_mode=label_mode,
                                                    n_bits=n_bits,
                                                    n_classes_c1=country_n,
                                                    n_classes_c2=country_n)
                        ).to(device)'''
    uw = UncertaintyWeighting().to(device)
    uw = None
    
    split = (0.9, 0.1)
    # Dataset(s)
    if label_mode == "ecoc":
        assert codes_path is not None, "codes_path is required for ECOC mode."
        codebook = load_ecoc_codes(codes_path)
        # Optional: verify bit length
        kb = next(iter(codebook.values()))
        assert len(kb) == n_bits, f"ECOC bits mismatch: got n_bits={n_bits}, codebook has {len(kb)}"
        train_ds = BordersParquet(parquet_path, split="train", split_frac=split, codebook=codebook, label_mode="ecoc")
        val_ds   = BordersParquet(parquet_path, split="val",   split_frac=split, codebook=codebook, label_mode="ecoc")
        class_cfg = ClassHeadConfig(class_mode="ecoc", n_bits=n_bits)
        print(f"ECOC: {n_bits} bits/head; codebook loaded from {codes_path}")
    elif label_mode == "softmax":
        codebook = None
        train_ds = BordersParquet(parquet_path, split="train", split_frac=split, codebook=None, label_mode="softmax")
        val_ds   = BordersParquet(parquet_path, split="val",   split_frac=split, codebook=None, label_mode="softmax")
        class_cfg = ClassHeadConfig(
            class_mode="softmax",
            n_classes_c1=train_ds.num_classes_c1,
            n_classes_c2=train_ds.num_classes_c2,
        )
    else:
        raise ValueError("label_mode must be 'ecoc' or 'softmax'")

    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    # Optim
    if uw:
        opt = torch.optim.AdamW(
        [{"params": model.parameters()},
        {"params": uw.parameters(), "weight_decay": 0.0}],
        lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    lw = LossWeights(*loss_weights)
    

    best_val = math.inf
    for ep in range(1, epochs+1):
        tr = train_one_epoch(model, train_loader, opt, device, lw, label_mode=class_cfg.class_mode, uw=uw)
        va = evaluate(model, val_loader, device, lw, label_mode=class_cfg.class_mode, codebook=codebook)
        line = (f"[{ep:02d}] train: {tr}  |  val: {va}")
        print(line)
        
        if va["rmse_km"] < best_val:
            best_val = va["rmse_km"]
            if label_mode == "ecoc":
                best_val + (1.0 - va["c1_decoded_acc"]) + (1.0 - va["c1_decoded_acc"])
            else: 
                best_val + (1.0 - va["c1_top1"]) + (1.0 - va["c2_top1"])

            ckpt = {
                "model": model.state_dict(),
                "uw": uw.state_dict() if uw else 1.0,
                "config": {
                    "label_mode": class_cfg.class_mode,
                    "n_bits": getattr(class_cfg, "n_bits", None),
                    "n_classes_c1": getattr(class_cfg, "n_classes_c1", None),
                    "n_classes_c2": getattr(class_cfg, "n_classes_c2", None),
                     "layers": layer_counts,
                },
            }
            out_path = pathlib.Path("python/nn_checkpoints")
            out_path.mkdir(parents=True, exist_ok=True)
            save_path = out_path / f"siren_{label_mode}_1M_6x256_fixed_lw.pt"
            torch.save(ckpt, save_path)
            print(f"  ↳ saved checkpoint: {save_path}")

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    PATH = "python/geodata/parquet/dataset_all.parquet"
    # weights found through uw
    dist_w = 0.27296819293 # 1/LARGEST_DIST**2
    c1_w = 1.78925619504
    c2_w = 1.55426070234
    
    # weights found by finding the practical max of each loss
    main(PATH, epochs=20, device=device, label_mode="ecoc",
         layer_counts=(256,)*6
         ,loss_weights=(dist_w, c1_w, c2_w)
         ,w0=30.0
         )


    