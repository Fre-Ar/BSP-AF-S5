from siren import SIRENLayer, SIREN
from nir import NIRLayer, NIRTrunk, MultiHeadNIR
import torch, torch.nn as nn, torch.nn.functional as F
import math 
from torch.utils.data import DataLoader
from data import BordersParquet, LossWeights, train_one_epoch, evaluate, load_ecoc_codes

class Tester:
    def __init__(self, start, activation, params): 
        super().__init__()
        self.start = start
        self.activation = activation
        self.params = params
        self.activation(self, ("start",), *self.params)

def linear(self, vars, a, b):
    result = a * getattr(self, vars[0]) + b
    setattr(self, vars[0], result)

#tester = Tester(3.14, linear, (2.0, 1.0))
#print(tester.start)


#w0 = 30.0
#layer_counts = (256,)*5
#siren = MultiHeadNIR(SIRENLayer, in_dim=3, layer_counts=layer_counts, params=(w0,)*5)


def test_SIREN_1D():
    # ---- Data: f(t) = sin(6t) + 0.5 sin(13t) ----
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    torch.manual_seed(0)

    N = 65536
    t = torch.rand(N, 1)*2 - 1                               # [-1,1]
    f = torch.sin(6*t) + 0.5*torch.sin(13*t)

    model = SIREN(in_dim=1, out_dim=1, hidden=64, depth=4, w0_first=30.0, w0_hidden=1.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    t, f = t.to(device), f.to(device)

    for step in range(900):
        idx = torch.randint(0, N, (2048,), device=device)
        pred = model(t[idx])
        loss = F.mse_loss(pred, f[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 150 == 0:
            print(f"step {step:4d} | mse {loss.item():.6e}")

    # quick sanity: print a few predictions
    with torch.no_grad():
        probe = torch.linspace(-1, 1, 9, device=device).unsqueeze(1)
        gt = torch.sin(6*probe)+0.5*torch.sin(13*probe)
        pr = model(probe)
        for x, g, y in zip(probe.flatten().tolist(), gt.flatten().tolist(), pr.flatten().tolist()):
            print(f"t={x:+.2f}  gt={g:+.3f}  pred={y:+.3f}")
            

def test_SIREN_2D():
    # ---- Data: f(t) = sin(6t) + 0.5 sin(13t) ----
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    torch.manual_seed(0)

    r = 0.6
    N = 131072
    xy = torch.rand(N, 2)*2 - 1     # [-1,1]^2
    dist = (xy.pow(2).sum(dim=1, keepdim=True).sqrt() - r).abs()

    model = SIREN(in_dim=2, out_dim=1, hidden=128, depth=4, w0_first=30.0, w0_hidden=1.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    xy, dist = xy.to(device), dist.to(device)

    softplus = nn.Softplus()
    for step in range(3000):
        idx = torch.randint(0, N, (4096,), device=device)
        pred = softplus(model(xy[idx]))  # keep nonnegative if you like
        loss = F.mse_loss(pred, dist[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 300 == 0:
            rmse = math.sqrt(loss.item())
            print(f"step {step:4d} | rmse {rmse:.6e}")

    # quick sanity: print a few predictions
    # Inspect along a radial line (y=0)
    with torch.no_grad():
        xline = torch.linspace(-1, 1, 11, device=device).unsqueeze(1)
        yzero = torch.zeros_like(xline)
        pts = torch.cat([xline, yzero], dim=1)
        gt = (pts.pow(2).sum(dim=1, keepdim=True).sqrt() - r).abs()
        pr = softplus(model(pts))
        for x, g, y in zip(xline.flatten().tolist(), gt.flatten().tolist(), pr.flatten().tolist()):
            print(f"x={x:+.2f}  gt={g:.3f}  pred={y:.3f}")
            
            

# ===================== MAIN =====================

def main(parquet_path, codes_path="python/geodata/countries.ecoc.json",
         batch_size=8192, epochs=10,
         layer_counts=(256,)*5, w0_first=30.0, w0_hidden=1.0,
         lr=9e-4, loss_weights=(1.0,1.0,1.0),
         n_bits=32,
         max_dist_km=None, device=None):

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    ecoc_codes = load_ecoc_codes(codes_path)  # id(int) -> ecoc(1xn_bits)
    
    # Datasets
    split = (0.9, 0.1)
    train_ds = BordersParquet(
        parquet_path, split="train", split_frac=split,
        codebook=ecoc_codes
    )
    val_ds = BordersParquet(
        parquet_path, split="val", split_frac=split,
        codebook=ecoc_codes
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size*2, shuffle=False, num_workers=0)

    # Model

    depth = len(layer_counts)
    model = MultiHeadNIR(SIRENLayer,
                         in_dim=3,
                         layer_counts=layer_counts,
                         params=((w0_first,),)+((w0_hidden,),)*(depth-1),
                         code_bits=n_bits
                         ).to(device)

    # Optim
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    lw = LossWeights(*loss_weights)
    print(f"Training on device: {device}")
    print(f"ECOC: {n_bits} bits/head; codebook loaded from {codes_path}")

    best = None
    for ep in range(1, epochs+1):
        tr = train_one_epoch(model, train_loader, opt, device, lw, max_dist_km=max_dist_km)
        va = evaluate(model, val_loader, device, lw, max_dist_km=max_dist_km)
        line = (f"[{ep:03d}] "
                f"train rmse={tr['rmse_km']:.3f}km c1={tr['c1_bit_acc']:.3f} c2={tr['c2_bit_acc']:.3f} | "
                f"val rmse={va['rmse_km']:.3f}km c1={va['c1_bit_acc']:.3f} c2={va['c2_bit_acc']:.3f}")
        print(line)
        # Track best by val RMSE + (1-acc) penalties
        score = va["rmse_km"] + (1.0 - va["c1_bit_acc"]) + (1.0 - va["c2_bit_acc"])
        if best is None or score < best[0]:
            best = (score, {k:float(v) for k,v in va.items()})
            torch.save({
                "model": model.state_dict(),
                "ecoc": {
                    "bits": n_bits,
                    "codes_path": codes_path 
                }
            }, "python/nn_checkpoints/siren_best.pt")
            print("  ↳ saved checkpoint: siren_best.pt")

if __name__ == "__main__":
    PATH = "python/geodata/parquet/dataset_all.parquet"
    main(PATH,
         epochs=20)
    