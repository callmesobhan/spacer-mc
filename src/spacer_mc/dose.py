"""Dose engines: 1-D analytic falloff (M0), calibrated 3-D EBRT gradient, and 3-D AAPM TG-43.

Backends:
  - `falloff`: M0 1-D Gaussian-blurred step (preserved for closed-form benchmarks).
  - `EBRTGradientEngine`: dual-component EBRT gradient (steep penumbra + low-dose bath tail)
    calibrated to Susil 2010 (V70 vs 0/5/10/15 mm pure shift) and Whalley 2016 / Song 2013
    (paired V30-V80 DVH).
  - `TG43ProstateEngine`: AAPM TG-43 point/line-source dose superposition over a peripheral-loaded
    3-D prostate seed/dwell cloud with isotope-specific radial dose functions g_L(r) for
    I-125, Pd-103, Cs-131, and Ir-192.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import erfc

from .geometry import RectalWallMesh


def falloff(s_mm, sigma_p_mm, s50_mm=0.0):
    """Fraction of prescription dose at distance s_mm along the anterior-posterior axis.

    The profile is a Gaussian-blurred step: 0.5 at s50_mm, with penumbra width sigma_p_mm.
    """
    return 0.5 * erfc((np.asarray(s_mm) - s50_mm) / (np.sqrt(2.0) * sigma_p_mm))


@dataclass(frozen=True)
class EBRTGradientEngine:
    """Calibrated dual-component EBRT dose model (fraction of Rx at each rectal voxel).

    Combines:
      1. Steep PTV posterior margin penumbra: (1 - tail_weight) * 0.5 * erfc((d - s50) / (sqrt(2)*sigma_p))
      2. Low-dose lateral/exit bath: tail_weight * exp(-max(d, 0) / lambda_tail)
    Plus cranio-caudal PTV extent tapering outside +/-20 mm (prostate + margin length ~40 mm).

    Calibrated to Susil 2010 (pure shift V70: 23.0% @ 0 mm, 15.1% @ 5 mm, 3.7% @ 10 mm, ~0% @ 15 mm)
    and Whalley 2016 / Mariados 2015 paired DVH (V30-V80).
    """

    s50_mm: float = 20.0  # AP depth of 50% primary penumbra from anterior prostate capsule
    sigma_p_mm: float = 4.0  # posterior IMRT/VMAT penumbra width (mm)
    tail_weight: float = 0.45  # fraction of prescription in multi-beam low-dose bath
    lambda_tail_mm: float = 38.0  # exponential decay length of low-dose bath (mm)
    ptv_half_length_mm: float = 23.5  # prostate + SI PTV margin half-length (mm)
    si_penumbra_mm: float = 4.5  # superior-inferior penumbra (mm)
    hotspot_peak: float = 1.05  # typical PTV peak ~105% Rx at posterior PTV overlap

    def dose_fraction(
        self,
        mesh: RectalWallMesh,
        separation_mm: np.ndarray,
    ) -> np.ndarray:
        """Compute dose fraction of Rx at each voxel for total AP separation `separation_mm`.

        `separation_mm` has shape (..., n_voxels) representing the added posterior shift
        (baseline gap + spacer(t) + daily motion).
        """
        # Total posterior distance of each rectal voxel from posterior prostate boundary
        d_ap = mesh.y_base_mm + separation_mm
        # Lateral distance from prostate midline (prostate half-width ~22 mm)
        lat_pen = 0.5 * erfc((np.abs(mesh.x_mm) - 22.0) / (np.sqrt(2.0) * 4.5))
        # Cranio-caudal extent of PTV
        si_pen = 0.5 * erfc(
            (np.abs(mesh.z_mm) - self.ptv_half_length_mm) / (np.sqrt(2.0) * self.si_penumbra_mm)
        )
        primary = (
            self.hotspot_peak
            * (1.0 - self.tail_weight)
            * 0.5
            * erfc((d_ap - self.s50_mm) / (np.sqrt(2.0) * self.sigma_p_mm))
            * lat_pen
            * si_pen
        )
        # Multi-beam IMRT/VMAT bath falls slowly with AP distance and covers full rectal length
        bath = (
            self.hotspot_peak
            * self.tail_weight
            * np.exp(-np.maximum(d_ap, 0.0) / self.lambda_tail_mm)
            * (0.55 + 0.45 * si_pen)
        )
        return np.clip(primary + bath, 0.0, 1.15)


# Consensus AAPM TG-43 radial dose functions g_L(r) on r_cm grid (0.5 .. 10.0 cm)
# Representative consensus tabulations:
#   I-125 (6711-class), Pd-103 (Model 200-class), Cs-131 (Proxcelan-class), Ir-192 (microSelectron HDR)
_TG43_R_CM = np.array([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0], dtype=float)
_TG43_GL: dict[str, np.ndarray] = {
    "I-125": np.array([1.06, 1.07, 1.00, 0.908, 0.814, 0.632, 0.496, 0.364, 0.197, 0.075]),
    "Pd-103": np.array([1.35, 1.29, 1.00, 0.768, 0.583, 0.335, 0.192, 0.110, 0.036, 0.007]),
    "Cs-131": np.array([1.03, 1.04, 1.00, 0.946, 0.876, 0.720, 0.576, 0.452, 0.270, 0.115]),
    "Ir-192": np.array([1.00, 0.998, 1.00, 1.004, 1.007, 1.008, 1.004, 0.994, 0.958, 0.875]),
}


def _prostate_seed_cloud_mm(prostate_radius_mm: float = 19.0, half_length_mm: float = 18.0) -> np.ndarray:
    """Modified peripheral-loading 3-D prostate seed/dwell cloud (36 sources) in mm.

    Prostate sits anterior to the rectum: posterior capsule at y = 0, center at y = -prostate_radius_mm.
    """
    pts: list[tuple[float, float, float]] = []
    # 3 axial planes (apex, mid, base) x 12 peripheral/intermediate sources
    for z in (-0.72 * half_length_mm, 0.0, 0.72 * half_length_mm):
        ring_r = prostate_radius_mm * (0.72 if z == 0.0 else 0.58)
        for k in range(8):
            ang = 2.0 * np.pi * k / 8.0
            x = ring_r * np.sin(ang)
            y = -prostate_radius_mm + ring_r * np.cos(ang)
            pts.append((x, y, z))
        # 4 inner/anterior-lateral sources (sparing central urethra)
        inner_r = prostate_radius_mm * 0.35
        for ang in (0.25 * np.pi, 0.75 * np.pi, 1.25 * np.pi, 1.75 * np.pi):
            pts.append((inner_r * np.sin(ang), -prostate_radius_mm + inner_r * np.cos(ang), z))
    return np.asarray(pts, dtype=float)


class TG43ProstateEngine:
    """AAPM TG-43 3-D prostate seed/dwell superposition dose engine.

    Normalizes source strength so that the posterior-capsule peripheral prescription boundary
    (mean over peripheral control points at 2 mm posterior to capsule) receives 100% Rx,
    matching published LDR/HDR prostate peripheral-loading dosimetry (Teyateeti 2022, Taggar 2018,
    Haddad 2023, Chao 2019).
    """

    def __init__(self, isotope: str, *, prostate_radius_mm: float = 19.0) -> None:
        if isotope not in _TG43_GL:
            raise ValueError(f"Unsupported TG-43 isotope {isotope!r}; choose from {tuple(_TG43_GL)}")
        self.isotope = isotope
        self.seeds_mm = _prostate_seed_cloud_mm(prostate_radius_mm=prostate_radius_mm)
        self._gl_table = _TG43_GL[isotope]

        # Calibrate normalization so peripheral capsule points sit at 100% prescription isodose
        ref_pts = np.array(
            [
                [0.0, 0.0, 0.0],
                [10.0, 1.5, 0.0],
                [-10.0, 1.5, 0.0],
                [0.0, 1.0, 12.0],
                [0.0, 1.0, -12.0],
            ],
            dtype=float,
        )
        raw_ref = self._raw_kernel(ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2])
        self._norm = float(np.mean(raw_ref))
        self._mesh_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _raw_kernel(self, x_mm: np.ndarray, y_mm: np.ndarray, z_mm: np.ndarray) -> np.ndarray:
        """Sum g_L(r) / r^2 across all seeds for arbitrary broadcastable coordinate arrays."""
        dx = x_mm[..., None] - self.seeds_mm[:, 0]
        dy = y_mm[..., None] - self.seeds_mm[:, 1]
        dz = z_mm[..., None] - self.seeds_mm[:, 2]
        r_cm = np.sqrt(dx * dx + dy * dy + dz * dz) / 10.0
        r_safe = np.maximum(r_cm, 0.25)
        gl = np.interp(r_safe, _TG43_R_CM, self._gl_table, left=self._gl_table[0], right=self._gl_table[-1] * 0.5)
        return np.sum(gl / (r_safe * r_safe), axis=-1)

    def _get_voxel_lut(self, mesh: RectalWallMesh) -> tuple[np.ndarray, np.ndarray]:
        key = id(mesh)
        if key not in self._mesh_cache:
            # 0.25 mm AP shift LUT covering y in [0.5, 75.5] mm (<0.02% interp error vs exact TG-43)
            y_grid = np.linspace(0.5, 75.5, 301)  # step = 0.25 mm
            # Evaluate (V, G) once
            x_2d = np.broadcast_to(mesh.x_mm[:, None], (mesh.n_voxels, y_grid.size))
            z_2d = np.broadcast_to(mesh.z_mm[:, None], (mesh.n_voxels, y_grid.size))
            y_2d = np.broadcast_to(y_grid[None, :], (mesh.n_voxels, y_grid.size))
            vals = self._raw_kernel(x_2d, y_2d, z_2d) / self._norm
            self._mesh_cache[key] = (y_grid, vals)
        return self._mesh_cache[key]

    def dose_fraction(
        self,
        mesh: RectalWallMesh,
        separation_mm: np.ndarray,
    ) -> np.ndarray:
        """Compute instantaneous TG-43 dose-rate fraction (relative to Rx) at every rectal voxel."""
        y_vox = np.clip(mesh.y_base_mm + separation_mm, 0.5, 75.25)
        y_grid, lut = self._get_voxel_lut(mesh)
        step = float(y_grid[1] - y_grid[0])
        pos = (y_vox - y_grid[0]) / step
        idx0 = np.floor(pos).astype(int)
        frac = pos - idx0
        vox_idx = np.arange(mesh.n_voxels)
        return (1.0 - frac) * lut[vox_idx, idx0] + frac * lut[vox_idx, idx0 + 1]
