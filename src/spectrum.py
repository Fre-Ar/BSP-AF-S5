# src/spectrum.py

import math
import pathlib
from typing import Tuple, Optional

import numpy as np
import torch
import os

from nirs.create_nirs import build_model
from nirs.nns.nir import LabelMode
from utils.utils import human_int, get_default_device
from utils.utils_geo import DATA_ANALYSIS_PATH


# ---------------------------------------------------------
# Core helpers
# ---------------------------------------------------------

def _build_architecture(
    model_name: str,
    layer_counts: Tuple[int, ...],
    label_mode: LabelMode,
    w0: float,
    w_hidden: float,
    s: float,
    beta: float,
    global_z: bool,
    encoder_params: Optional[Tuple[float, float, float]],
    device: str,
) -> torch.nn.Module:
    """
    Thin wrapper around nirs.create_nirs.build_model, reusing your existing
    factory and keeping the exact calling convention used in training.py.
    """
    params = (w0, w_hidden, s, beta, global_z)
    model, _ = build_model(
        model_name,
        layer_counts,
        label_mode,
        params,
        encoder_params,
    )
    return model.to(device)


@torch.no_grad()
def _probe_single_network(
    model: torch.nn.Module,
    in_dim: int,
    n_samples: int,
    device: str,
    out_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Probe a single randomly initialized network along a 1D line x(t),
    compute FFT magnitude, and return (freqs, mag).

    - We sample t in [-1, 1] with uniform spacing.
    - We choose a random direction d in R^{in_dim} and evaluate x(t) = t * d.
    - For MultiHeadNIR, we take the first output (distance head).
    """

    # 1D parameter
    t = torch.linspace(-1.0, 1.0, n_samples, device=device)  # (N,)
    dt = float(t[1] - t[0])

    # Random direction in input space
    d = torch.randn(in_dim, device=device)
    d = d / d.norm()

    # Points along the line
    x = t.unsqueeze(1) * d.unsqueeze(0)  # (N, in_dim)

    # Forward pass
    out = model(x)

    # Handle different model return types
    if isinstance(out, (tuple, list)):
        # MultiHeadNIR returns (dist, c1_logits, c2_logits)
        y = out[0]  # distance head
    else:
        y = out

    y = y.reshape(-1).detach().cpu().numpy()
    y = y - y.mean()  # remove DC component

    # FFT magnitude
    Y = np.fft.rfft(y)
    mag = np.abs(Y)

    # Frequency axis (in "cycles per unit t")
    freqs = np.fft.rfftfreq(n_samples, d=dt)

    return freqs, mag

@torch.no_grad()
def _probe_single_network_on_s2(        # <<< CHANGED FOR S^2: new function name & behavior
    model: torch.nn.Module,
    n_samples: int,
    device: str,
    out_index: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Probe a single randomly initialized network along a GREAT CIRCLE on S^2,   # <<< CHANGED FOR S^2
    compute FFT magnitude, and return (freqs, mag).

    - We sample θ in [-π, π] with uniform spacing.
    - We choose a random orthonormal pair (u, v) in ℝ^3 and define:
        x(θ) = cos θ * u + sin θ * v   (points on the unit sphere).
    """

    # Parameter along the great circle (angle in radians).
    theta = torch.linspace(-math.pi, math.pi, n_samples, device=device)  # (N,)
    dtheta = float(theta[1] - theta[0])

    # Random orthonormal frame (u, v) spanning a great circle on S^2.
    u = torch.randn(3, device=device)
    u = u / u.norm()

    v = torch.randn(3, device=device)
    v = v - torch.dot(v, u) * u  # make v orthogonal to u
    v = v / v.norm()

    # Points on the unit sphere along the great circle.
    cos_t = torch.cos(theta).unsqueeze(1)
    sin_t = torch.sin(theta).unsqueeze(1)
    x = cos_t * u.unsqueeze(0) + sin_t * v.unsqueeze(0)  # (N, 3), |x|=1

    # Forward pass
    out = model(x)

    # Handle different model return types (MultiHeadNIR etc.)
    if isinstance(out, (tuple, list)):
        # MultiHeadNIR returns (dist, c1_logits, c2_logits)
        y = out[0]  # distance head
    else:
        y = out

    y = y.reshape(-1).detach().cpu().numpy()
    y = y - y.mean()  # remove DC component

    # FFT magnitude
    Y = np.fft.rfft(y)
    mag = np.abs(Y)

    # Frequency axis (cycles per radian along θ).
    freqs = np.fft.rfftfreq(n_samples, d=dtheta)

    return freqs, mag

def _moving_average(x: np.ndarray, window: int = 31) -> np.ndarray:
    """
    Simple 1D moving average smoother for the intrinsic spectrum.
    """
    window = max(3, window | 1)  # enforce odd window >= 3
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def estimate_intrinsic_cutoff(
    model_name: str,
    layer_counts: Tuple[int, ...],
    label_mode: LabelMode,
    w0: float,
    w_hidden: float,
    s: float,
    beta: float,
    global_z: bool,
    encoder_params: Optional[Tuple[float, float, float]],
    input_dim: int = 3,
    n_networks: int = 16,
    n_samples: int = 8192,
    device: Optional[str] = None,
    deriv_threshold: float = 6e-4,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Main routine: follow the efficient-sampling paper to estimate the cut-off frequency.

    Returns
    -------
    F_c : float
        Estimated cut-off frequency.
    freqs : np.ndarray
        Frequency grid (same length as E).
    E_smooth : np.ndarray
        Smoothed intrinsic spectrum (averaged magnitude over random nets).
    """
    if device is None:
        device = (
            torch.accelerator.current_accelerator().type
            if torch.accelerator.is_available()
            else "cpu"
        )
    print(f"[intrinsic_spectrum] Using device: {device}")

    # Accumulate spectra
    avg_mag = None
    freqs_ref = None

    for i in range(n_networks):
        torch.manual_seed(1234 + i)
        model = _build_architecture(
            model_name=model_name,
            layer_counts=layer_counts,
            label_mode=label_mode,
            w0=w0,
            w_hidden=w_hidden,
            s=s,
            beta=beta,
            global_z=global_z,
            encoder_params=encoder_params,
            device=device,
        )
        model.eval()

        freqs, mag = _probe_single_network(
            model=model,
            in_dim=input_dim,
            n_samples=n_samples,
            device=device,
        )

        if avg_mag is None:
            avg_mag = np.zeros_like(mag, dtype=np.float64)
            freqs_ref = freqs
        else:
            assert np.allclose(freqs_ref, freqs), "Frequency grids mismatch."

        avg_mag += mag

    avg_mag /= float(n_networks)

    # Smooth the intrinsic spectrum (paper uses a rational fit; we approximate
    # this with a moving average for robustness).
    E = avg_mag
    E_smooth = _moving_average(E, window=31)

    # Numerical derivative dE/dF
    dE_dF = np.gradient(E_smooth, freqs_ref)

    # Ignore DC
    mask = freqs_ref > 0.0

    # Find first frequency where the spectrum has "flattened":
    # |dE/dF| < deriv_threshold, following the spirit of C'(F_c) = 6e-4 in the paper.
    candidates = np.where(mask & (np.abs(dE_dF) < deriv_threshold))[0]
    if len(candidates) == 0:
        F_c = float(freqs_ref[-1])
        print(
            "[intrinsic_spectrum] Warning: derivative threshold not reached; "
            "using highest available frequency as F_c."
        )
    else:
        F_c = float(freqs_ref[candidates[0]])

    return F_c, freqs_ref, E_smooth

def estimate_intrinsic_cutoff_on_s2(    # <<< CHANGED FOR S^2: renamed + S^2 semantics
    model_name: str,
    layer_counts: Tuple[int, ...],
    label_mode: str,
    w0: float,
    w_hidden: float,
    s: float,
    beta: float,
    global_z: bool,
    encoder_params: Optional[Tuple[float, float, float]],
    n_networks: int = 16,
    n_samples: int = 8192,
    device: Optional[str] = None,
    deriv_threshold: float = 6e-4,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Main routine: follow the efficient-sampling paper to estimate the cut-off
    frequency F_c, but adapted to the S^2 domain.

    Returns
    -------
    F_c : float
        Estimated cut-off frequency (cycles per radian along a great circle).
    freqs : np.ndarray
        Frequency grid (same length as E).
    E_smooth : np.ndarray
        Smoothed intrinsic spectrum (averaged magnitude over random nets).
    """
    # Device selection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[intrinsic_spectrum] Using device: {device}")

    # Accumulate spectra
    avg_mag = None
    freqs_ref = None

    for i in range(n_networks):
        torch.manual_seed(1234 + i)
        model = _build_architecture(
            model_name=model_name,
            layer_counts=layer_counts,
            label_mode=label_mode,
            w0=w0,
            w_hidden=w_hidden,
            s=s,
            beta=beta,
            global_z=global_z,
            encoder_params=encoder_params,
            device=device,
        )
        model.eval()

        freqs, mag = _probe_single_network_on_s2(   # <<< CHANGED FOR S^2: call S^2 probe
            model=model,
            n_samples=n_samples,
            device=device,
        )

        if avg_mag is None:
            avg_mag = np.zeros_like(mag, dtype=np.float64)
            freqs_ref = freqs
        else:
            assert np.allclose(freqs_ref, freqs), "Frequency grids mismatch."

        avg_mag += mag

    avg_mag /= float(n_networks)

    # Smooth the intrinsic spectrum
    E = avg_mag
    E_smooth = _moving_average(E, window=31)

    # Numerical derivative dE/dF
    dE_dF = np.gradient(E_smooth, freqs_ref)

    # Ignore DC
    mask = freqs_ref > 0.0

    # Find first frequency where the spectrum has "flattened":
    # |dE/dF| < deriv_threshold, following the spirit of C'(F_c) = 6e-4.
    candidates = np.where(mask & (np.abs(dE_dF) < deriv_threshold))[0]
    if len(candidates) == 0:
        F_c = float(freqs_ref[-1])
        print(
            "[intrinsic_spectrum] Warning: derivative threshold not reached; "
            "using highest available frequency as F_c."
        )
    else:
        F_c = float(freqs_ref[candidates[0]])

    return F_c, freqs_ref, E_smooth

def main() -> None:
    MODEL = "siren"
    MODE = "ecoc" 
    DEPTH = 12
    LAYER = 128
    LAYER_COUNTS = (LAYER,)*DEPTH

    W0 = 30.0 
    WH = 1.0
    S = 1.0
    BETA = 1.0
    GLOBAL_Z = True

    input_dim=3
    encoder_params=(16, 2.0 * math.pi, 1.0)
    
    n_networks=16
    n_samples=8192
    device=get_default_device()
    deriv_threshold=6e-4
    
    out_npz=None #os.path.join(DATA_ANALYSIS_PATH, f"{MODEL}_{MODE}_1M_{DEPTH}x{LAYER}_w0{W0}_wh{WH}.pt" ),

    F_c, freqs, E_smooth = estimate_intrinsic_cutoff_on_s2(
        model_name=MODEL,
        layer_counts=LAYER_COUNTS,
        
        w0=W0,
        w_hidden=WH,
        s=S,
        beta=BETA,
        global_z=GLOBAL_Z,
        encoder_params=encoder_params,
        
        label_mode=MODE,
        
        n_networks=n_networks,
        n_samples=n_samples,
        device=device,
        deriv_threshold=deriv_threshold,
    )

    # --- Nyquist-based recommended sampling for S^2 --------------------------
    # Domain intrinsic dimension is 2 (sphere surface), so we use:
    #   μ_2D ≈ (2 F_c)^2   samples per unit area.
    # The unit sphere S^2 has area 4π, so total samples:
    #   N_sphere ≈ μ_2D * 4π.                                             
    # ------------------------------------------------------------------------
    mu_2d = (2.0 * F_c) ** 2
    total_sphere_samples = mu_2d * (4.0 * math.pi)

    print("\n=== Intrinsic spectrum summary ===")
    print(f"  model_name     : {MODEL}")
    print(f"  layer_counts   : {LAYER_COUNTS}")
    print(f"  label_mode     : {MODE}")
    print(f"  w0 / w_hidden  : {W0} / {WH}")
    print(f"  s, beta        : {S}, {BETA}")
    print(f"  global_z       : {GLOBAL_Z}")
    print(f"  encoder_params : {encoder_params}")
    print(f"  n_networks     : {n_networks}")
    print(f"  n_samples(line): {human_int(n_samples)}")
    print(f"\n  Estimated cut-off F_c ≈ {F_c:.4f} cycles / radian (great circle)")
    print(f"  2D Nyquist density μ_2D = (2 F_c)^2 ≈ {mu_2d:.3e} samples / unit area on S^2")
    print(
        f"  Total Nyquist samples on S^2 ≈ μ_2D * 4π ≈ "
        f"{total_sphere_samples:.3e}  (~{human_int(int(total_sphere_samples))} points)"
    )
    
    if out_npz is not None:
        out_path = pathlib.Path(out_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_path,
            freqs=freqs,
            E_smooth=E_smooth,
            F_c=F_c,
            mu_2d=mu_2d,
            total_sphere_samples=total_sphere_samples,
            meta=dict(
                model_name=MODEL,
                layer_counts=LAYER_COUNTS,
                label_mode=MODE,
                w0=W0,
                w_hidden=WH,
                s=S,
                beta=BETA,
                global_z=GLOBAL_Z,
                encoder_params=encoder_params,
                n_networks=n_networks,
                n_samples=n_samples,
                domain="S^2",
            ),
        )
        print(f"\n  Saved spectrum to: {out_path}")


if __name__ == "__main__":
    main()
