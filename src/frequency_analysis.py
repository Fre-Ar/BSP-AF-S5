# src/frequency_analysis.py

from __future__ import annotations
import math
from typing import Tuple

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from geodata.sampler.sampling import BorderSampler
from utils.utils_geo import (
    R_EARTH_KM,
    BORDERS_FGB_PATH,
    arc_segment_attrs,
    normalize_vec,
    move_along_geodesic,
    unitvec_to_lonlat,
    SEED
)
from utils.utils import human_int

# ---------- random directions on the sphere ----------

def _random_unitvec(rng: np.random.Generator) -> np.ndarray:
    """
    Draw a random unit vector on the sphere.
    Uses a Gaussian in R^3 followed by normalization.
    """
    v = rng.normal(size=(1, 3))
    v = normalize_vec(v)
    return v[0].astype(np.float32)


def _random_tangent_vec(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Draw a random unit tangent vector at point p on the sphere.
    This is orthogonal to p and normalized.
    """
    v = rng.normal(size=(1, 3))
    # Project to tangent plane at p
    proj = v - np.sum(v * p[None, :], axis=-1, keepdims=True) * p[None, :]
    proj = normalize_vec(proj)
    return proj[0].astype(np.float32)


# ---------- sampling along a great circle ----------

def sample_geodesic_distances(
    sampler: BorderSampler,
    p0: np.ndarray,
    t0: np.ndarray,
    num_samples: int,
) -> Tuple[np.ndarray, float]:
    """
    Sample the ground-truth distance field along a full great circle.

    Parameters
    ----------
    sampler : BorderSampler
        Ground-truth labeller (distance + country IDs).
    p0 : ndarray, shape (3,)
        Base point on the sphere, unit vector.
    t0 : ndarray, shape (3,)
        Tangent direction at p0, unit vector orthogonal to p0.
    num_samples : int
        Number of samples along the great circle.

    Returns
    -------
    d_km : ndarray, shape (num_samples,)
        Distances to nearest border (km) along the geodesic.
    length_km : float
        Physical length of the great circle (2π R_EARTH_KM).
    """
    theta = np.linspace(0.0, 2.0 * math.pi, num_samples, endpoint=False, dtype=np.float64)

    # Broadcast p0 and t0 to match theta
    p = np.repeat(p0[None, :], num_samples, axis=0)
    t = np.repeat(t0[None, :], num_samples, axis=0)

    # Move along geodesic and convert to lon/lat for the sampler
    xyz = move_along_geodesic(p, t, theta)  # (N,3), already normalized
    lon_deg, lat_deg = unitvec_to_lonlat(xyz)  # each shape (N,)

    d_km = np.empty(num_samples, dtype=np.float32)
    for i in range(num_samples):
        d, _, _ = sampler.sample_lonlat(float(lon_deg[i]), float(lat_deg[i]))
        d_km[i] = d

    length_km = 2.0 * math.pi * R_EARTH_KM
    return d_km, length_km


# ---------- 1D spectrum ----------

def power_spectrum_1d(signal: np.ndarray, length_km: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D power spectrum of a real signal sampled uniformly along a curve.

    Parameters
    ----------
    signal : ndarray, shape (N,)
        Signal samples along the curve.
    length_km : float
        Physical length of the curve (in km).

    Returns
    -------
    freqs : ndarray, shape (N//2+1,)
        Frequency bins in cycles per km.
    power : ndarray, shape (N//2+1,)
        Power per frequency bin (|FFT|^2 / N).
    """
    N = signal.shape[0]
    sig_centered = signal - signal.mean()

    fft = np.fft.rfft(sig_centered)
    power = (np.abs(fft) ** 2) / N

    dx = length_km / N
    freqs = np.fft.rfftfreq(N, d=dx)  # cycles per km
    return freqs, power


# ---------- main estimator ----------

def estimate_effective_fmax(
    sampler: BorderSampler,
    num_geodesics: int = 8,
    num_samples: int = 8192,
    energy_threshold: float = 0.99,
    mode: str = "border_kernel",
    sigma_km: float = 10.0,
    d_max_km: float = 100.0,
    seed: int = 42,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Empirically estimate an effective maximum spatial frequency f_max of the field.

    Modes
    -----
    - "global":
        Use the raw distance d(x) along full great circles (coarse, interior-dominated).
    - "border_kernel" (default):
        Use a border-emphasizing signal g(d) = exp(-d / sigma_km),
        optionally zeroed out for d > d_max_km. This focuses the spectrum
        on near-border regions.

    Parameters
    ----------
    sampler : BorderSampler
        Ground-truth labeller.
    num_geodesics : int
        Number of random great circles to sample.
    num_samples : int
        Points per geodesic (resolution; smaller dx = 2πR / num_samples).
    energy_threshold : float
        Fraction of total non-DC energy below f_max (e.g. 0.99).
    mode : {"global", "border_kernel"}
        Spectrum mode (see above).
    sigma_km : float
        Length scale for the border kernel in km (used in "border_kernel" mode).
    d_max_km : float
        Hard cutoff for distances in "border_kernel" mode (points with d > d_max_km
        are set to 0).
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    f_eff : float
        Effective f_max in cycles/km.
    freqs : ndarray
        Frequency bins (cycles/km).
    mean_power : ndarray
        Mean power spectrum over all sampled geodesics.
    """
    mode = mode.lower()
    assert mode in {"global", "border_kernel"}

    rng = np.random.default_rng(seed)

    freqs = None
    accum_power = None

    for g in range(num_geodesics):
        p0 = _random_unitvec(rng)
        t0 = _random_tangent_vec(p0, rng)

        d_km, length_km = sample_geodesic_distances(
            sampler, p0, t0, num_samples=num_samples
        )

        if mode == "global":
            signal = d_km
        else:
            # border_kernel: emphasize near borders
            # g(d) = exp(-d / sigma), zero out far points
            signal = np.exp(-d_km / sigma_km)
            if d_max_km is not None:
                mask_far = d_km > d_max_km
                signal[mask_far] = 0.0

        f, power = power_spectrum_1d(signal, length_km)

        if freqs is None:
            freqs = f
            accum_power = power
        else:
            if f.shape != freqs.shape or not np.allclose(f, freqs):
                raise RuntimeError("Frequency grids differ between geodesics.")
            accum_power += power

    mean_power = accum_power / float(num_geodesics)

    # Ignore DC when computing cumulative energy
    power_no_dc = mean_power.copy()
    power_no_dc[0] = 0.0
    total_energy = power_no_dc.sum()

    if total_energy <= 0:
        raise RuntimeError("Total non-DC energy is zero or negative; check data / mode.")

    cumsum = np.cumsum(power_no_dc)
    idx = int(np.searchsorted(cumsum, energy_threshold * total_energy))
    idx = min(idx, freqs.shape[0] - 1)

    f_eff = float(freqs[idx])
    return f_eff, freqs, mean_power




def compute_total_border_length_km(borders_path: str | None = None) -> float:
    """
    Compute total length of all border segments in km.

    If the FlatGeobuf already has a 'theta_ab' column (as produced by
    preprocess_borders.create_borders), we just use that. Otherwise we
    recompute the spherical arc angles with arc_segment_attrs.
    """
    if borders_path is None:
        borders_path = BORDERS_FGB_PATH

    gdf = gpd.read_file(borders_path)

    if "theta_ab" in gdf.columns:
        # theta_ab is already the short-arc angle in radians
        theta_ab = gdf["theta_ab"].to_numpy(dtype=np.float64)
    else:
        # Fallback: recompute from lon/lat endpoints
        # (this should normally not be needed for your current pipeline)
        geom = gdf.geometry.values
        ax = np.array([seg.coords[0][0] for seg in geom], dtype=np.float64)
        ay = np.array([seg.coords[0][1] for seg in geom], dtype=np.float64)
        bx = np.array([seg.coords[-1][0] for seg in geom], dtype=np.float64)
        by = np.array([seg.coords[-1][1] for seg in geom], dtype=np.float64)

        A3, B3, N3, theta_ab, M3, mask_valid = arc_segment_attrs(
            ax, ay, bx, by, min_arc_deg=None
        )
        # In case mask_valid filters anything out
        theta_ab = theta_ab[mask_valid]

    # Great-circle length of each sub-segment in km
    seg_len_km = theta_ab * R_EARTH_KM
    total_km = float(seg_len_km.sum())
    return total_km


def border_length():
    spacing_km = 5.0

    total_km = compute_total_border_length_km(BORDERS_FGB_PATH)
    n_points = total_km / spacing_km if spacing_km > 0 else float("inf")

    print("\n=== Border length estimate ===")
    print(f"Borders file          : {BORDERS_FGB_PATH}")
    print(f"Total border length   : {total_km:,.1f} km "
          f"(~{human_int(int(round(total_km)))} km)")
    print(f"Target spacing        : {spacing_km:.3f} km")
    print(f"Points needed (1D)    : {n_points:,.0f} "
          f"(~{human_int(int(round(n_points)))})")
    print("  (This is per 'line', ignoring radial / multi-band duplication.)\n")




def main():
    sampler = BorderSampler()  # Uses default GPKG_PATH / BORDERS_FGB_PATH / etc.

    energy_threshold = 0.999
    mode ="global"
    #mode="border_kernel"
    
    f_eff, freqs, mean_power = estimate_effective_fmax(
        sampler,
        num_geodesics=8,
        num_samples=8192,
        energy_threshold=energy_threshold,
        seed=SEED,
        mode=mode,
        sigma_km=10.0,
        d_max_km=50.0,
    )

    nyquist_spacing_km = 1.0 / (2.0 * f_eff) if f_eff > 0 else float("inf")

    print("")
    print("=== Distance field bandwidth estimate ===")
    print(f"Mode                   : {mode}")
    print(f"Energy threshold       : {energy_threshold:.3f}")
    print(f"Effective f_max        : {f_eff:.6f} cycles/km")
    print(f"Nyquist spacing (km)   : {nyquist_spacing_km:.3f} km")
    print("")
    
    plot = True

    if plot:
        # Log-log plot of the averaged spectrum
        eps = 1e-12
        plt.figure(figsize=(7, 4), dpi=120)
        plt.loglog(freqs[1:], mean_power[1:] + eps, label="Mean spectrum (no DC)")
        plt.axvline(
            f_eff,
            color="red",
            linestyle="--",
            label=f"f_max (≈ {f_eff:.3g} cycles/km)",
        )
        plt.xlabel("Spatial frequency [cycles/km]")
        plt.ylabel("Power")
        plt.title(f"Distance field spectrum — mode={mode}")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    #border_length()
    main()
    #print("hi")
