"""3-D anterior-to-posterior rectal wall geometry, per-level spacer profiles, and heavy-tailed motion.

Grounded in published morphology and motion priors from data/priors.yaml:
  - Per-level base > mid > apex thickness profile: Pinkawa 2013, Song 2013, Choi 2026, Prusty 2025.
  - Lateral gel taper beyond +/-10 mm: Hwang 2018.
  - Placement failure / apex absence tail (~8.8%): Ben-Dor 2025, Fischer-Valuck 2017.
  - Heavy-tailed daily separation noise: Polizzi 2024 (Student-t, df=6, scale=3.7 mm matches
    77.2% within +/-5 mm and 3.6% beyond +/-10 mm).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RectalWallMesh:
    """3-D rectal cylinder/wall sample grid for exact DVH, V70, D2cc, D1cc, and D0.1cc metrics.

    Coordinate frame (mm, centered at mid-gland posterior prostate surface = y=0):
      x: left-right lateral axis
      y: anterior-posterior distance posteriorly from posterior prostate capsule (y >= 0)
      z: cranio-caudal axis (-18 mm apex to +18 mm base)
    """

    x_mm: np.ndarray  # (V,)
    y_base_mm: np.ndarray  # (V,) baseline AP depth from prostate surface into rectum (without spacer)
    z_mm: np.ndarray  # (V,)
    voxel_vol_cc: float  # volume per voxel in cm^3
    total_vol_cc: float  # total rectal volume (~45 cc standard prostate planning rectum)
    # Precomputed cranio-caudal basis weights for (apex, mid, base) interpolation, shape (3, V)
    level_weights: np.ndarray
    # Precomputed lateral Gaussian gel-coverage factor (Hwang 2018: falls beyond 10 mm lateral)
    lateral_factor: np.ndarray

    @property
    def n_voxels(self) -> int:
        return int(self.x_mm.size)


def make_rectal_mesh(
    total_vol_cc: float = 45.0,
    radius_mm: float = 15.0,
    length_mm: float = 54.0,
    n_z: int = 9,
    n_theta: int = 16,
    n_r: int = 4,
) -> RectalWallMesh:
    """Build a lightweight 3-D cylindrical rectal volume (~576 voxels, fast vectorized evaluation).

    A 54 mm long, 15 mm radius rectum gives V ~ 38-45 cc (scaled cleanly to total_vol_cc).
    Anterior rectal wall touches y = 0 at theta = 0; posterior wall sits at y = 2 * radius_mm.
    """
    zs = np.linspace(-0.5 * length_mm, 0.5 * length_mm, n_z)
    thetas = np.linspace(-np.pi, np.pi, n_theta, endpoint=False)
    # Radial shells from outer wall inward (equal area weighting via sqrt spacing)
    r_edges = np.sqrt(np.linspace(0.0, 1.0, n_r + 1)) * radius_mm
    r_mids = 0.5 * (r_edges[:-1] + r_edges[1:])

    Z, T, R = np.meshgrid(zs, thetas, r_mids, indexing="ij")
    x = (R * np.sin(T)).ravel()
    # Anterior wall (cos(T)=1, R=radius) sits at y=0; center at y=radius; posterior at y=2*radius.
    y = (radius_mm - R * np.cos(T)).ravel()
    z = Z.ravel()

    # Cranio-caudal piecewise-linear weights for apex (-15), mid (0), base (+15)
    z_clamped = np.clip(z, -15.0, 15.0)
    w_apex = np.clip(-z_clamped / 15.0, 0.0, 1.0)
    w_base = np.clip(z_clamped / 15.0, 0.0, 1.0)
    w_mid = 1.0 - np.abs(z_clamped) / 15.0
    level_weights = np.vstack([w_apex, w_mid, w_base])

    # Lateral gel falloff: full thickness at midline, tapering past +/-10 mm (Hwang 2018)
    lateral_factor = np.exp(-0.5 * (x / 10.5) ** 2)

    voxel_vol = total_vol_cc / x.size
    return RectalWallMesh(
        x_mm=x,
        y_base_mm=y,
        z_mm=z,
        voxel_vol_cc=voxel_vol,
        total_vol_cc=total_vol_cc,
        level_weights=level_weights,
        lateral_factor=lateral_factor,
    )


@dataclass(frozen=True)
class SpacerProfiles:
    """Per-patient cranio-caudal thickness (apex, mid, base) + lateral asymmetry (shape (N,))."""

    apex_mm: np.ndarray
    mid_mm: np.ndarray
    base_mm: np.ndarray
    asymmetry_mm: np.ndarray  # lateral center shift (+right / -left)

    def thickness_on_mesh(self, mesh: RectalWallMesh) -> np.ndarray:
        """Spatial spacer separation s0(patient, voxel) in mm, shape (n_patients, n_voxels)."""
        # Cranio-caudal thickness profile: (N, 3) @ (3, V) -> (N, V)
        levels = np.column_stack([self.apex_mm, self.mid_mm, self.base_mm])
        cc_thick = levels @ mesh.level_weights
        # Lateral coverage shifted by patient asymmetry (Hwang 2018: falls beyond 10 mm lateral)
        lat = np.exp(-0.5 * ((mesh.x_mm[None, :] - self.asymmetry_mm[:, None]) / 10.5) ** 2)
        return np.clip(cc_thick * lat, 0.0, None)


def fixed_patient_profile(
    apex_mm: float,
    mid_mm: float,
    base_mm: float,
    n_patients: int,
    asymmetry_mm: float = 0.0,
) -> SpacerProfiles:
    """Create a constant patient-specific spacer profile replicated across MC draws."""
    return SpacerProfiles(
        apex_mm=np.full(n_patients, max(0.0, float(apex_mm)), dtype=float),
        mid_mm=np.full(n_patients, max(0.0, float(mid_mm)), dtype=float),
        base_mm=np.full(n_patients, max(0.0, float(base_mm)), dtype=float),
        asymmetry_mm=np.full(n_patients, float(asymmetry_mm), dtype=float),
    )


def sample_population_profiles(
    material: str,
    n_patients: int,
    rng: np.random.Generator,
    *,
    mid_override_mm: float | None = None,
    include_placement_failures: bool = True,
) -> SpacerProfiles:
    """Sample realistic base > mid > apex spacer profiles (Pinkawa 2013, Mariados 2015, Choi 2026).

    If `mid_override_mm` is given, pins achieved mid-gland separation for inverse-separation sweeps
    while preserving realistic apex (~0.82x) and base (~1.12x) anatomical tapering.
    """
    if material == "none":
        zeros = np.zeros(n_patients, dtype=float)
        return SpacerProfiles(zeros, zeros, zeros, zeros)

    if mid_override_mm is not None:
        mid = np.full(n_patients, max(0.0, float(mid_override_mm)), dtype=float)
        apex = np.clip(mid * 0.82, 0.0, None)
        base = np.clip(mid * 1.12, 0.0, None)
        asym = np.zeros(n_patients, dtype=float)
        return SpacerProfiles(apex, mid, base, asym)

    # Material-specific baseline achieved separation (mm) from published cohorts
    mean_mid, sd_mid = {
        "PEG": (11.5, 3.6),  # Mariados 2015 (12.6-1.6=11.0 gain), Miller 2020 (11.2), Choi 2026 (12.0)
        "HA": (10.5, 3.2),  # Bjoreland 2023 / Agrawal 2026 (~9.2-11.0 mm)
        "DuraSeal": (10.5, 3.4),  # Vaggers 2021
        "Balloon": (15.0, 2.8),  # Agrawal 2026 (15.5 mm)
    }[material]

    mid = np.clip(rng.normal(mean_mid, sd_mid, size=n_patients), 0.0, 25.0)
    # Correlated base > mid > apex profile (Pinkawa 2013: 16/12/10 mm; Choi 2026: 12/12/11 mm)
    apex = np.clip(mid * rng.normal(0.82, 0.16, size=n_patients), 0.0, 25.0)
    base = np.clip(mid * rng.normal(1.12, 0.16, size=n_patients), 0.0, 28.0)
    asym = rng.normal(0.0, 3.5, size=n_patients)  # lateral offset SD ~3.5 mm

    if include_placement_failures:
        # Ben-Dor 2025: 8.8% no spacer at apex; Mariados 2015 / Miller 2020: ~2% complete failure
        apex_absent = rng.random(n_patients) < 0.088
        apex = np.where(apex_absent, 0.0, apex)
        complete_fail = rng.random(n_patients) < 0.02
        mid = np.where(complete_fail, 0.0, mid)
        apex = np.where(complete_fail, 0.0, apex)
        base = np.where(complete_fail, 0.0, base)

    return SpacerProfiles(apex_mm=apex, mid_mm=mid, base_mm=base, asymmetry_mm=asym)


def sample_daily_motion_mm(
    n_patients: int,
    n_times: int,
    rng: np.random.Generator,
    *,
    with_spacer: bool = False,
    scale_mm: float = 3.7,
    df: float = 6.0,
    sys_sd_mm: float = 1.6,
    deterministic: bool = False,
) -> np.ndarray:
    """Sample heavy-tailed anterior-posterior prostate-rectum separation shifts (mm).

    Positive shift moves rectum AWAY from prostate (larger gap); negative shift moves rectum
    CLOSER to prostate (higher rectal dose).
    Polizzi 2024: Student-t(df=6, scale=3.7 mm) matches 77.2% within +/-5 mm and 3.6% beyond +/-10 mm.
    Pinkawa 2013: hydrogel damps posterior prostate systematic displacement (27% >6.5 mm -> 0%).
    """
    if deterministic:
        return np.zeros((n_patients, n_times), dtype=float)
    eff_sys_sd = sys_sd_mm * (0.65 if with_spacer else 1.0)
    eff_rand_scale = scale_mm * (0.75 if with_spacer else 1.0)
    systematic = rng.normal(0.0, eff_sys_sd, size=(n_patients, 1))
    # Slight upward delivered-dose bias (mean -0.5 mm closer in controls; Patrick 2024 / Petrikowski 2025)
    drift = -0.5 if not with_spacer else -0.15
    random_t = rng.standard_t(df=df, size=(n_patients, n_times)) * eff_rand_scale
    return np.clip(drift + systematic + random_t, -18.0, 18.0)
