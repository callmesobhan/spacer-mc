"""Rectal DVH extraction, EQD2 conversion, and LKB NTCP radiobiology (EBRT only).

Honoring DESIGN.md §3 & §6:
  - LKB NTCP is used for photon EBRT only (never transferred to LDR permanent implants).
  - Default QUANTEC late Grade >=2 rectal toxicity parameters (Michalski et al. 2010):
    TD50 = 76.9 Gy, m = 0.13, n = 0.09, alpha/beta = 3.0 Gy.
  - Tested against Wong 2025 (17 studies, 3,200 men): late Grade >=2 rectal toxicity
    RR = 0.25 (95% CI 0.15-0.42), 1.62% vs 9.35%.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def v_dose_pct(doses_gy: np.ndarray, threshold_gy: float) -> np.ndarray:
    """Percentage of rectal volume receiving >= threshold_gy, shape (n_patients,)."""
    return 100.0 * np.mean(doses_gy >= threshold_gy, axis=-1)


def v_dose_cc(doses_gy: np.ndarray, threshold_gy: float, voxel_vol_cc: float) -> np.ndarray:
    """Absolute rectal volume (cc) receiving >= threshold_gy, shape (n_patients,)."""
    return np.sum(doses_gy >= threshold_gy, axis=-1) * float(voxel_vol_cc)


def d_xcc(doses_gy: np.ndarray, volume_cc: float, voxel_vol_cc: float) -> np.ndarray:
    """Minimum dose to the hottest `volume_cc` of rectum (e.g. D2cc, D1cc, D0.1cc), shape (n_patients,)."""
    n_vox = doses_gy.shape[-1]
    k = max(1, min(n_vox, round(volume_cc / voxel_vol_cc)))
    # Partition from the high end: index n_vox - k is the k-th hottest voxel
    idx = n_vox - k
    part = np.partition(doses_gy, idx, axis=-1)
    return part[..., idx]


def eqd2(dose_gy: np.ndarray, n_fractions: int, alpha_beta: float = 3.0) -> np.ndarray:
    """Equivalent total dose in 2 Gy fractions (LQ model; late rectal alpha/beta = 3 Gy)."""
    d_fx = np.asarray(dose_gy, dtype=float) / float(n_fractions)
    return dose_gy * (d_fx + alpha_beta) / (2.0 + alpha_beta)


def lkb_ntcp(
    eqd2_gy: np.ndarray,
    *,
    td50_gy: float = 76.9,
    m: float = 0.13,
    n: float = 0.09,
) -> np.ndarray:
    """Lyman-Kutcher-Burman NTCP via generalized equivalent uniform dose (gEUD).

    `eqd2_gy` has shape (..., n_voxels) with equal-volume voxels.
    Returns NTCP probability in [0, 1] of shape (...).
    """
    d = np.maximum(np.asarray(eqd2_gy, dtype=float), 0.0)
    inv_n = 1.0 / n
    # Log-sum-exp style stabilization to prevent overflow for small n (e.g. n=0.07-0.09)
    d_max = np.maximum(np.max(d, axis=-1, keepdims=True), 1e-12)
    scaled = d / d_max
    geud = d_max[..., 0] * np.power(np.mean(np.power(scaled, inv_n), axis=-1), n)
    t = (geud - td50_gy) / (m * td50_gy)
    return 0.5 * (1.0 + erf(t / np.sqrt(2.0)))
