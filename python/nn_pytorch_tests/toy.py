# python/nn_pytorch_tests/toy.py

from nn_siren import SIRENLayer, SIREN
from nir import NIRLayer, NIRTrunk, MultiHeadNIR, ClassHeadConfig
import torch, torch.nn as nn, torch.nn.functional as F
import math 
from torch.utils.data import DataLoader
from data import BordersParquet, LossWeights, train_one_epoch, evaluate, load_ecoc_codes, BIT_LENGTH    
import pathlib

# ===================== MAIN =====================

def main(parquet_path, codes_path="python/geodata/countries.ecoc.json",
         batch_size=8192, epochs=10,
         layer_counts=(256,)*5, w0_first=30.0, w0_hidden=1.0,
         lr=9e-4, loss_weights=(1.0,1.0,1.0),
         n_bits=BIT_LENGTH,
         label_mode: str = "ecoc",  # 'ecoc' or 'softmax'
         device: str | None = None):

    print(f"Training on device: {device}")
    
    # Model
    depth = len(layer_counts)
    model = MultiHeadNIR(SIRENLayer,
                         in_dim=3,
                         layer_counts=layer_counts,
                         params=((w0_first,),)+((w0_hidden,),)*(depth-1),
                         code_bits=n_bits
                         ).to(device)
    
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
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lw = LossWeights(*loss_weights)
    

    best_val = math.inf
    for ep in range(1, epochs+1):
        tr = train_one_epoch(model, train_loader, opt, device, lw, label_mode=class_cfg.class_mode)
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
            save_path = out_path / "siren_best.pt"
            torch.save(ckpt, save_path)
            print(f"  ↳ saved checkpoint: {save_path}")

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    PATH = "python/geodata/parquet/dataset_all.parquet"
    main(PATH, epochs=20, device=device)


    