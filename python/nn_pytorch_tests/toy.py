# python/nn_pytorch_tests/toy.py

from nn_siren import SIRENLayer, Sine
from nn_relu import ReLULayer
from fourier_features import PositionalEncoding
from nir import NIRLayer, NIRTrunk, MultiHeadNIR, ClassHeadConfig
import torch, torch.nn as nn, torch.nn.functional as F
import math 
from torch.utils.data import DataLoader
from data import BordersParquet, LossWeights, train_one_epoch, evaluate, load_ecoc_codes, BIT_LENGTH    
import pathlib
from loss import UncertaintyWeighting
from stats import ecoc_prevalence_by_bit, pos_weight_from_prevalence
from time import perf_counter

# ===================== MAIN =====================

def main(parquet_path, codes_path="python/geodata/countries.ecoc.json",
         batch_size=8192, epochs=10,
         layer_counts=(256,)*5, w0=30.0, w_hidden=1.0,
         lr=9e-4, loss_weights=(1.0,1.0,1.0),
         n_bits=BIT_LENGTH,
         label_mode: str = "ecoc",  # 'ecoc' or 'softmax'
         country_n: int = 289,
         device: str | None = None,
         debug_losses: bool = False,
         head_layers: tuple = ()):

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
                         ,head_layers=head_layers
                         #,head_activation=Sine(w_hidden)
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
    
    if label_mode == "ecoc":
        # BCE bit prevalence correction
        ones_c1, totals_c1, p1_c1 = ecoc_prevalence_by_bit(parquet_path, codebook, id_col="c1_id")
        ones_c2, totals_c2, p1_c2 = ecoc_prevalence_by_bit(parquet_path, codebook, id_col="c2_id")

        pw_c1 = pos_weight_from_prevalence(p1_c1)
        pw_c2 = pos_weight_from_prevalence(p1_c2)
    else:
        pw_c1 = pw_c2 = None

        '''tensor([0.3267, 2.4615, 3.0240, 2.4913, 0.3675, 3.1064, 2.7537, 0.3025, 0.2411,
        0.2572, 2.9166, 3.2772, 0.3784, 0.3891, 2.6629, 0.3960, 2.7765, 0.3878,
        0.3001, 0.2895, 0.3536, 3.4143, 0.4478, 2.2867, 3.9613, 0.4065, 4.0929,
        0.2964, 0.2939, 4.1599, 0.2736, 3.7409]) 
        
        tensor([0.5199, 1.5317, 1.9189, 1.6731, 0.6476, 1.8805, 1.7705, 0.4535, 0.3375,
        0.4466, 1.8992, 2.1045, 0.6381, 0.6166, 1.6940, 0.6423, 1.5000, 0.5638,
        0.5068, 0.5126, 0.4965, 2.1715, 0.6538, 1.5002, 2.4251, 0.6675, 2.2548,
        0.5003, 0.5436, 2.2663, 0.4827, 2.2470])'''
    
    best_val = math.inf
    for ep in range(1, epochs+1):
        tr = train_one_epoch(
            model, train_loader, opt, device, lw,
            pos_weight_c1=pw_c1,
            pos_weight_c2=pw_c2,
            label_mode=class_cfg.class_mode,
            uw=uw,
            debug_losses=debug_losses)
        va = evaluate(
            model, val_loader, device, lw,
            pos_weight_c1=pw_c1,
            pos_weight_c2=pw_c2,
            label_mode=class_cfg.class_mode,
            codebook=codebook)
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
            
            save_path = out_path / f"siren_{label_mode}_1M_{depth}x{layer_counts[0]}_{len(head_layers)}h_w{w0}_post.pt"
            torch.save(ckpt, save_path)
            print(f"  ↳ saved checkpoint: {save_path}")

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    PATH = "python/geodata/parquet/log_dataset_1M.parquet"
    # weights found through uw
    dist_w = 0.27296819293 
    c1_w = 1.78925619504
    c2_w = 1.55426070234
    
    dist_w = 1/500**2 # 1/LARGEST_MSE**2
    c1_w = 1.0
    c2_w = 1.1765
    
    dist_w = 0.125
    dist_w = 0.5
    c1_w = 1.0
    c2_w = 1.0
    
    
    # weights found by finding the practical max of each loss
    main(PATH, epochs=20, device=device, label_mode="ecoc",
         layer_counts=(128,)*15
         #,loss_weights=(dist_w, c1_w, c2_w)
         ,w0=30.0
         #,debug_losses = True
         #,head_layers=(128,)
         )


    