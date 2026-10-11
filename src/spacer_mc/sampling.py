"""Patient and fraction sampler (M0: one point on the anterior rectal wall, 1-D geometry)."""

from dataclasses import dataclass

import numpy as np

from .dose import falloff


@dataclass(frozen=True)
class Scenario:
    """All lengths in mm. Defaults are illustrative placeholders, not literature values."""

    gap_mm: float = 2.0  # wall distance from the 50% isodose with no spacer
    spacer_mean_mm: float = 10.0  # mean achieved spacer thickness
    spacer_sd_mm: float = 2.0  # patient-to-patient spread in achieved thickness
    sigma_p_mm: float = 3.0  # dose penumbra width
    sigma_sys_mm: float = 1.5  # per-patient systematic rectal offset (same every fraction)
    sigma_rand_mm: float = 2.5  # fraction-to-fraction rectal motion
    n_fractions: int = 20


def simulate(sc: Scenario, n_patients: int, seed: int):
    """Mean wall dose fraction per patient, without and with spacer.

    Both arms reuse the same sampled offsets and motion (common random numbers), so the
    difference between them has far lower variance than two independent runs.
    """
    rng = np.random.default_rng(seed)
    thickness = np.clip(rng.normal(sc.spacer_mean_mm, sc.spacer_sd_mm, n_patients), 0.0, None)
    systematic = rng.normal(0.0, sc.sigma_sys_mm, (n_patients, 1))
    motion = rng.normal(0.0, sc.sigma_rand_mm, (n_patients, sc.n_fractions))

    offset = systematic + motion
    d_without = falloff(sc.gap_mm + offset, sc.sigma_p_mm).mean(axis=1)
    d_with = falloff(sc.gap_mm + thickness[:, None] + offset, sc.sigma_p_mm).mean(axis=1)
    return d_without, d_with
