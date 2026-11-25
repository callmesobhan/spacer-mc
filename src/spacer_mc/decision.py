"""4D coupled hydrogel-resorption x isotope-decay x motion simulator and inverse separation solver.

Commercial treatment-planning systems (Eclipse, RayStation, VariSeed, Oncentra Brachy) evaluate
post-injection spacer geometry as frozen in time. This module integrates:
  1. Material-specific 4D separation resorption kinetics (PEG vs HA vs DuraSeal vs Balloon),
  2. Modality/isotope delivery clocks (I-125 vs Pd-103 vs Cs-131 vs Ir-192 vs EBRT/SBRT),
  3. Post-implant LDR prostate oedema decay (Haddad 2023), and
  4. Heavy-tailed inter-fraction rectal motion (Polizzi 2024)
to quantify the *Hidden Resorption Penalty* (4D delivered minus static planning dose) and solve
the inverse clinical problem: *what minimum achieved post-injection separation gives 95% confidence
of meeting a rectal constraint after resorption and motion?*
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from .dose import EBRTGradientEngine, TG43ProstateEngine
from .geometry import (
    RectalWallMesh,
    SpacerProfiles,
    fixed_patient_profile,
    make_rectal_mesh,
    sample_daily_motion_mm,
    sample_population_profiles,
)
from .kinetics import (
    ISOTOPES,
    MATERIALS,
    decay_quadrature,
    fraction_schedule_days,
    oedema_contraction_mm,
    protected_dose_fraction,
    sample_kinetics_params,
    spacer_retention,
)
from .radbio import d_xcc, eqd2, lkb_ntcp, v_dose_cc, v_dose_pct


@dataclass(frozen=True)
class RegimeSpec:
    """Clinical radiotherapy regime specification with primary rectal constraint."""

    id: str
    label: str
    modality: str  # ebrt | ldr | hdr | ldr_ebrt_boost
    rx_gy: float
    n_fractions: int  # 0 for continuous LDR decay
    isotope: str | None  # I-125, Pd-103, Cs-131, Ir-192, or None (EBRT)
    start_day: float  # days from spacer injection to first fraction / implant
    fx_per_week: float
    constraint_metric: str  # e.g. V70_pct, V57_pct, D1cc_gy, D2cc_pct_rx, V75_cc
    constraint_limit: float
    constraint_unit: str
    constraint_source: str
    ebrt_boost_rx_gy: float = 0.0
    ebrt_boost_start_day: float = 56.0
    ebrt_boost_fractions: int = 25


REGIMES: dict[str, RegimeSpec] = {
    "ebrt_conv": RegimeSpec(
        id="ebrt_conv",
        label="EBRT Conventional (78 Gy / 39 fx, 8 wk)",
        modality="ebrt",
        rx_gy=78.0,
        n_fractions=39,
        isotope=None,
        start_day=10.0,
        fx_per_week=5.0,
        constraint_metric="V70_pct",
        constraint_limit=10.0,
        constraint_unit="%",
        constraint_source="Pederson / QUANTEC (rV70 <= 10%)",
    ),
    "ebrt_hypo": RegimeSpec(
        id="ebrt_hypo",
        label="EBRT Moderate Hypofractionation (60 Gy / 20 fx, 4 wk)",
        modality="ebrt",
        rx_gy=60.0,
        n_fractions=20,
        isotope=None,
        start_day=10.0,
        fx_per_week=5.0,
        constraint_metric="V57_pct",
        constraint_limit=10.0,
        constraint_unit="%",
        constraint_source="CHHiP / PROFIT (rV57 <= 10%)",
    ),
    "ebrt_sbrt": RegimeSpec(
        id="ebrt_sbrt",
        label="Prostate SBRT (36.25 Gy / 5 fx, 2 wk)",
        modality="ebrt",
        rx_gy=36.25,
        n_fractions=5,
        isotope=None,
        start_day=10.0,
        fx_per_week=2.5,
        constraint_metric="D1cc_gy",
        constraint_limit=35.0,
        constraint_unit="Gy",
        constraint_source="PACE-B / NRG GU005 (D1cc <= 35 Gy)",
    ),
    "ldr_i125": RegimeSpec(
        id="ldr_i125",
        label="LDR Permanent Implant I-125 (145 Gy, T1/2 = 59.4 d)",
        modality="ldr",
        rx_gy=145.0,
        n_fractions=0,
        isotope="I-125",
        start_day=0.0,
        fx_per_week=0.0,
        constraint_metric="D2cc_pct_rx",
        constraint_limit=60.0,
        constraint_unit="% Rx",
        constraint_source="ABS / King 2021 (rectal D2cc <= 60% Rx ~ 87 Gy)",
    ),
    "ldr_pd103": RegimeSpec(
        id="ldr_pd103",
        label="LDR Permanent Implant Pd-103 (125 Gy, T1/2 = 17.0 d)",
        modality="ldr",
        rx_gy=125.0,
        n_fractions=0,
        isotope="Pd-103",
        start_day=0.0,
        fx_per_week=0.0,
        constraint_metric="D2cc_pct_rx",
        constraint_limit=60.0,
        constraint_unit="% Rx",
        constraint_source="ABS / King 2021 (rectal D2cc <= 60% Rx ~ 75 Gy)",
    ),
    "ldr_cs131": RegimeSpec(
        id="ldr_cs131",
        label="LDR Permanent Implant Cs-131 (115 Gy, T1/2 = 9.7 d)",
        modality="ldr",
        rx_gy=115.0,
        n_fractions=0,
        isotope="Cs-131",
        start_day=0.0,
        fx_per_week=0.0,
        constraint_metric="D2cc_pct_rx",
        constraint_limit=60.0,
        constraint_unit="% Rx",
        constraint_source="ABS / King 2021 (rectal D2cc <= 60% Rx ~ 69 Gy)",
    ),
    "ldr_i125_ebrt_boost": RegimeSpec(
        id="ldr_i125_ebrt_boost",
        label="I-125 LDR (110 Gy) + Delayed EBRT (45 Gy / 25 fx starting day 56; Kubo 2021)",
        modality="ldr_ebrt_boost",
        rx_gy=110.0,
        n_fractions=25,
        isotope="I-125",
        start_day=0.0,
        fx_per_week=5.0,
        constraint_metric="D2cc_gy",
        constraint_limit=105.0,
        constraint_unit="Gy",
        constraint_source="Combined LDR+EBRT rectal D2cc <= 105 Gy (Kubo 2021 timing hazard)",
        ebrt_boost_rx_gy=45.0,
        ebrt_boost_start_day=56.0,
        ebrt_boost_fractions=25,
    ),
    "hdr_ir192": RegimeSpec(
        id="hdr_ir192",
        label="HDR Ir-192 Boost (15 Gy x 1 fx)",
        modality="hdr",
        rx_gy=15.0,
        n_fractions=1,
        isotope="Ir-192",
        start_day=7.0,
        fx_per_week=1.0,
        constraint_metric="V75_cc",
        constraint_limit=1.0,
        constraint_unit="cc",
        constraint_source="RTOG 0321 / Chao 2019 (rectal V75% < 1.0 cc)",
    ),
}


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Exact Wilson 95% confidence interval for a binomial pass probability."""
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)
    return (max(0.0, center - margin), min(1.0, center + margin))


def _summary_stats(arr: np.ndarray) -> dict[str, float]:
    a = np.asarray(arr, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "sd": float(np.std(a, ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)),
        "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)),
        "mc_se": float(np.std(a, ddof=1) / np.sqrt(a.size)) if a.size > 1 else 0.0,
    }


def _extract_primary_metric(
    doses_gy: np.ndarray,
    regime: RegimeSpec,
    mesh: RectalWallMesh,
) -> np.ndarray:
    m = regime.constraint_metric
    if m == "V70_pct":
        return v_dose_pct(doses_gy, 70.0)
    if m == "V57_pct":
        return v_dose_pct(doses_gy, 57.0)
    if m == "D1cc_gy":
        return d_xcc(doses_gy, 1.0, mesh.voxel_vol_cc)
    if m == "D2cc_gy":
        return d_xcc(doses_gy, 2.0, mesh.voxel_vol_cc)
    if m == "D2cc_pct_rx":
        return 100.0 * d_xcc(doses_gy, 2.0, mesh.voxel_vol_cc) / regime.rx_gy
    if m == "V75_cc":
        return v_dose_cc(doses_gy, 0.75 * regime.rx_gy, mesh.voxel_vol_cc)
    raise ValueError(f"Unsupported primary metric {m!r}")


@dataclass(frozen=True)
class Simulation4DResult:
    """Full paired 4D simulation outcome comparing no-spacer, static TPS, and 4D delivered dose."""

    regime: RegimeSpec
    material: str
    n_patients: int
    seed: int
    metric_name: str
    metric_unit: str
    constraint_limit: float
    # Per-patient primary metric arrays
    metric_no_spacer: np.ndarray
    metric_static_tps: np.ndarray
    metric_4d_delivered: np.ndarray
    # Hidden resorption + motion penalty (4D delivered minus static frozen-geometry TPS)
    resorption_penalty: np.ndarray
    # Fraction of prescription dose delivered while spacer >= 5 mm
    protected_fraction: np.ndarray
    # Pass probabilities + Wilson 95% CIs
    pass_prob_no_spacer: float
    pass_prob_static_tps: float
    pass_prob_4d: float
    pass_ci95_4d: tuple[float, float]
    # Secondary EBRT LKB NTCP (None for brachytherapy per DESIGN.md)
    ntcp_no_spacer: np.ndarray | None
    ntcp_4d: np.ndarray | None
    ntcp_risk_ratio: float | None
    # Spearman variance attribution (input -> squared rank correlation share)
    sensitivity: dict[str, float]
    # Clinical decision flag
    risk_flag: str  # GREEN (>=95% pass) | AMBER (80-95%) | RED (<80%)
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_id": self.regime.id,
            "regime_label": self.regime.label,
            "material": self.material,
            "n_patients": self.n_patients,
            "seed": self.seed,
            "constraint": {
                "metric": self.metric_name,
                "limit": self.constraint_limit,
                "unit": self.metric_unit,
                "source": self.regime.constraint_source,
            },
            "no_spacer": _summary_stats(self.metric_no_spacer),
            "static_tps": _summary_stats(self.metric_static_tps),
            "delivered_4d": _summary_stats(self.metric_4d_delivered),
            "hidden_resorption_penalty": _summary_stats(self.resorption_penalty),
            "protected_dose_fraction": _summary_stats(self.protected_fraction),
            "pass_probability": {
                "no_spacer": round(self.pass_prob_no_spacer, 4),
                "static_tps": round(self.pass_prob_static_tps, 4),
                "delivered_4d": round(self.pass_prob_4d, 4),
                "delivered_4d_ci95": [round(self.pass_ci95_4d[0], 4), round(self.pass_ci95_4d[1], 4)],
            },
            "ntcp": None
            if self.ntcp_4d is None or self.ntcp_no_spacer is None
            else {
                "no_spacer_mean_pct": round(100.0 * float(np.mean(self.ntcp_no_spacer)), 3),
                "delivered_4d_mean_pct": round(100.0 * float(np.mean(self.ntcp_4d)), 3),
                "risk_ratio": round(float(self.ntcp_risk_ratio), 4) if self.ntcp_risk_ratio is not None else None,
            },
            "variance_attribution": {k: round(v, 4) for k, v in self.sensitivity.items()},
            "risk_flag": self.risk_flag,
            "recommendation": self.recommendation,
        }


def _evaluate_arm_ebrt(
    engine: EBRTGradientEngine,
    mesh: RectalWallMesh,
    rx_gy: float,
    baseline_gap_mm: np.ndarray,
    spacer_0_mm: np.ndarray,
    retention_t: np.ndarray,
    motion_t_mm: np.ndarray,
    time_weights: np.ndarray,
) -> np.ndarray:
    """Integrate EBRT voxel doses over treatment fractions, returning shape (n_patients, n_voxels)."""
    n_p = baseline_gap_mm.shape[0]
    acc = np.zeros((n_p, mesh.n_voxels), dtype=float)
    for k, wk in enumerate(time_weights):
        sep_k = baseline_gap_mm[:, None] + spacer_0_mm * retention_t[:, k : k + 1] + motion_t_mm[:, k : k + 1]
        acc += float(wk) * engine.dose_fraction(mesh, sep_k)
    return acc * rx_gy


def _evaluate_arm_tg43(
    engine: TG43ProstateEngine,
    mesh: RectalWallMesh,
    rx_gy: float,
    baseline_gap_mm: np.ndarray,
    spacer_0_mm: np.ndarray,
    retention_t: np.ndarray,
    oedema_t_mm: np.ndarray,
    motion_t_mm: np.ndarray,
    time_weights: np.ndarray,
) -> np.ndarray:
    """Integrate TG-43 voxel doses over radioactive decay / HDR fractions, shape (N, V)."""
    n_p = baseline_gap_mm.shape[0]
    acc = np.zeros((n_p, mesh.n_voxels), dtype=float)
    for k, wk in enumerate(time_weights):
        sep_k = (
            baseline_gap_mm[:, None]
            + spacer_0_mm * retention_t[:, k : k + 1]
            - oedema_t_mm[:, k : k + 1]
            + motion_t_mm[:, k : k + 1]
        )
        acc += float(wk) * engine.dose_fraction(mesh, sep_k)
    return acc * rx_gy


def _spearman_shares(target: np.ndarray, predictors: dict[str, np.ndarray]) -> dict[str, float]:
    """Normalized squared Spearman rank correlation attribution across sampled inputs."""
    raw: dict[str, float] = {}
    for name, vals in predictors.items():
        if np.std(vals) < 1e-12 or np.std(target) < 1e-12:
            raw[name] = 0.0
            continue
        rho = float(spearmanr(vals, target).statistic)
        raw[name] = 0.0 if math.isnan(rho) else rho * rho
    total = sum(raw.values())
    if total <= 0.0:
        return {k: 0.0 for k in raw}
    return {k: v / total for k, v in raw.items()}


def simulate_4d(
    regime_id: str,
    material: str = "PEG",
    *,
    n_patients: int = 1000,
    seed: int = 42,
    patient_profile_mm: tuple[float, float, float] | None = None,
    asymmetry_mm: float = 0.0,
    mid_override_mm: float | None = None,
    include_motion: bool = True,
    include_oedema: bool = True,
    include_placement_failures: bool = True,
    mesh: RectalWallMesh | None = None,
) -> Simulation4DResult:
    """Run paired 4D Monte Carlo propagation for a regime and hydrogel material.

    Compares three arms under common random numbers:
      1. `no_spacer`: anatomical baseline gap + motion + oedema (if LDR)
      2. `static_tps`: planning-day frozen spacer geometry (what commercial TPSs compute)
      3. `delivered_4d`: true time-integrated dose across resorption x decay clock x oedema x motion
    """
    if regime_id not in REGIMES:
        raise ValueError(f"Unknown regime {regime_id!r}; choose from {tuple(REGIMES)}")
    if material not in MATERIALS:
        raise ValueError(f"Unknown material {material!r}; choose from {MATERIALS}")

    regime = REGIMES[regime_id]
    mesh = mesh or make_rectal_mesh()
    ss = np.random.SeedSequence(seed)
    rng_geom, rng_kin, rng_mot, rng_oed = [np.random.default_rng(s) for s in ss.spawn(4)]

    # Baseline prostate-rectum gap without spacer: ~1.6-2.5 mm (Mariados 2015: 1.6 +/- 2.0 mm)
    if patient_profile_mm is not None or mid_override_mm is not None:
        base_gap = np.full(n_patients, 1.8, dtype=float)
    else:
        base_gap = np.clip(rng_geom.normal(1.8, 1.2, size=n_patients), 0.0, 8.0)

    if patient_profile_mm is not None:
        profiles: SpacerProfiles = fixed_patient_profile(
            *patient_profile_mm, n_patients=n_patients, asymmetry_mm=asymmetry_mm
        )
    else:
        profiles = sample_population_profiles(
            material,
            n_patients,
            rng_geom,
            mid_override_mm=mid_override_mm,
            include_placement_failures=include_placement_failures,
        )
    spacer_0 = profiles.thickness_on_mesh(mesh)  # (N, V)
    zeros_spacer = np.zeros_like(spacer_0)

    kin = sample_kinetics_params(material, n_patients, rng_kin)

    # Time schedules & dose engines per modality
    ebrt_engine = EBRTGradientEngine()
    oedema_amp = np.zeros(n_patients, dtype=float)

    if regime.modality == "ebrt":
        t_days, t_weights = fraction_schedule_days(
            regime.n_fractions, start_day=regime.start_day, fx_per_week=regime.fx_per_week
        )
        ret_t = spacer_retention(t_days, kin)
        ret_static = np.ones_like(ret_t)
        mot_ctrl = sample_daily_motion_mm(
            n_patients, t_days.size, rng_mot, with_spacer=False, deterministic=not include_motion
        )
        mot_sp = sample_daily_motion_mm(
            n_patients, t_days.size, rng_mot, with_spacer=(material != "none"), deterministic=not include_motion
        )
        zero_mot = np.zeros_like(mot_ctrl)

        dose_no = _evaluate_arm_ebrt(ebrt_engine, mesh, regime.rx_gy, base_gap, zeros_spacer, ret_static, mot_ctrl, t_weights)
        dose_static = _evaluate_arm_ebrt(ebrt_engine, mesh, regime.rx_gy, base_gap, spacer_0, ret_static, zero_mot, t_weights)
        dose_4d = _evaluate_arm_ebrt(ebrt_engine, mesh, regime.rx_gy, base_gap, spacer_0, ret_t, mot_sp, t_weights)
        prot = protected_dose_fraction(t_days, t_weights, profiles.mid_mm, kin)

    elif regime.modality == "ldr":
        assert regime.isotope is not None
        iso = ISOTOPES[regime.isotope]
        tg43 = TG43ProstateEngine(regime.isotope)
        t_days, t_weights = decay_quadrature(iso.half_life_days, n_steps=48)
        ret_t = spacer_retention(t_days, kin)
        ret_static = np.ones_like(ret_t)
        # Haddad 2023: day 1 -> day 30 oedema resolution shifts prostate sources closer by ~1.8 +/- 0.6 mm
        oedema_amp = (
            np.clip(rng_oed.normal(1.8, 0.6, size=n_patients), 0.0, 4.0)
            if include_oedema
            else np.zeros(n_patients)
        )
        oed_t = oedema_contraction_mm(t_days, oedema_amp, t_half_days=10.0)
        zero_oed = np.zeros_like(oed_t)
        # Continuous LDR averages daily random fluctuations; systematic offset + attenuated daily residual remain
        mot_ctrl = sample_daily_motion_mm(
            n_patients, t_days.size, rng_mot, with_spacer=False, scale_mm=1.8, sys_sd_mm=1.2, deterministic=not include_motion
        )
        mot_sp = sample_daily_motion_mm(
            n_patients, t_days.size, rng_mot, with_spacer=(material != "none"), scale_mm=1.8, sys_sd_mm=1.2, deterministic=not include_motion
        )
        zero_mot = np.zeros_like(mot_ctrl)

        dose_no = _evaluate_arm_tg43(tg43, mesh, regime.rx_gy, base_gap, zeros_spacer, ret_static, oed_t, mot_ctrl, t_weights)
        dose_static = _evaluate_arm_tg43(tg43, mesh, regime.rx_gy, base_gap, spacer_0, ret_static, zero_oed, zero_mot, t_weights)
        dose_4d = _evaluate_arm_tg43(tg43, mesh, regime.rx_gy, base_gap, spacer_0, ret_t, oed_t, mot_sp, t_weights)
        prot = protected_dose_fraction(t_days, t_weights, profiles.mid_mm, kin)

    elif regime.modality == "ldr_ebrt_boost":
        # Kubo 2021 combined I-125 LDR + delayed EBRT starting ~day 56 when PEG resorption begins!
        assert regime.isotope is not None
        iso = ISOTOPES[regime.isotope]
        tg43 = TG43ProstateEngine(regime.isotope)
        t_ldr, w_ldr = decay_quadrature(iso.half_life_days, n_steps=44)
        t_ebrt, w_ebrt = fraction_schedule_days(
            regime.ebrt_boost_fractions, start_day=regime.ebrt_boost_start_day, fx_per_week=5.0
        )
        ret_ldr = spacer_retention(t_ldr, kin)
        ret_ebrt = spacer_retention(t_ebrt, kin)
        oedema_amp = (
            np.clip(rng_oed.normal(1.8, 0.6, size=n_patients), 0.0, 4.0)
            if include_oedema
            else np.zeros(n_patients)
        )
        oed_ldr = oedema_contraction_mm(t_ldr, oedema_amp, t_half_days=10.0)
        mot_ldr = sample_daily_motion_mm(
            n_patients, t_ldr.size, rng_mot, with_spacer=(material != "none"), scale_mm=1.8, sys_sd_mm=1.2, deterministic=not include_motion
        )
        mot_ebrt = sample_daily_motion_mm(
            n_patients, t_ebrt.size, rng_mot, with_spacer=(material != "none"), deterministic=not include_motion
        )
        mot_ebrt_ctrl = sample_daily_motion_mm(
            n_patients, t_ebrt.size, rng_mot, with_spacer=False, deterministic=not include_motion
        )

        dose_no = (
            _evaluate_arm_tg43(tg43, mesh, regime.rx_gy, base_gap, zeros_spacer, np.ones_like(ret_ldr), oed_ldr, mot_ldr, w_ldr)
            + _evaluate_arm_ebrt(ebrt_engine, mesh, regime.ebrt_boost_rx_gy, base_gap, zeros_spacer, np.ones_like(ret_ebrt), mot_ebrt_ctrl, w_ebrt)
        )
        dose_static = (
            _evaluate_arm_tg43(tg43, mesh, regime.rx_gy, base_gap, spacer_0, np.ones_like(ret_ldr), np.zeros_like(oed_ldr), np.zeros_like(mot_ldr), w_ldr)
            + _evaluate_arm_ebrt(ebrt_engine, mesh, regime.ebrt_boost_rx_gy, base_gap, spacer_0, np.ones_like(ret_ebrt), np.zeros_like(mot_ebrt), w_ebrt)
        )
        dose_4d = (
            _evaluate_arm_tg43(tg43, mesh, regime.rx_gy, base_gap, spacer_0, ret_ldr, oed_ldr, mot_ldr, w_ldr)
            + _evaluate_arm_ebrt(ebrt_engine, mesh, regime.ebrt_boost_rx_gy, base_gap, spacer_0, ret_ebrt, mot_ebrt, w_ebrt)
        )
        # Weighted average protected dose fraction across LDR + delayed EBRT components
        f_ldr = regime.rx_gy / (regime.rx_gy + regime.ebrt_boost_rx_gy)
        prot = f_ldr * protected_dose_fraction(t_ldr, w_ldr, profiles.mid_mm, kin) + (1.0 - f_ldr) * protected_dose_fraction(t_ebrt, w_ebrt, profiles.mid_mm, kin)

    elif regime.modality == "hdr":
        tg43 = TG43ProstateEngine("Ir-192")
        t_days, t_weights = fraction_schedule_days(
            regime.n_fractions, start_day=regime.start_day, fx_per_week=regime.fx_per_week
        )
        ret_t = spacer_retention(t_days, kin)
        ret_static = np.ones_like(ret_t)
        zero_oed = np.zeros((n_patients, t_days.size))
        mot_ctrl = sample_daily_motion_mm(
            n_patients, t_days.size, rng_mot, with_spacer=False, scale_mm=1.2, sys_sd_mm=0.8, deterministic=not include_motion
        )
        mot_sp = sample_daily_motion_mm(
            n_patients, t_days.size, rng_mot, with_spacer=(material != "none"), scale_mm=1.2, sys_sd_mm=0.8, deterministic=not include_motion
        )
        zero_mot = np.zeros_like(mot_ctrl)

        dose_no = _evaluate_arm_tg43(tg43, mesh, regime.rx_gy, base_gap, zeros_spacer, ret_static, zero_oed, mot_ctrl, t_weights)
        dose_static = _evaluate_arm_tg43(tg43, mesh, regime.rx_gy, base_gap, spacer_0, ret_static, zero_oed, zero_mot, t_weights)
        dose_4d = _evaluate_arm_tg43(tg43, mesh, regime.rx_gy, base_gap, spacer_0, ret_t, zero_oed, mot_sp, t_weights)
        prot = protected_dose_fraction(t_days, t_weights, profiles.mid_mm, kin)
    else:
        raise ValueError(f"Unsupported modality {regime.modality!r}")

    m_no = _extract_primary_metric(dose_no, regime, mesh)
    m_static = _extract_primary_metric(dose_static, regime, mesh)
    m_4d = _extract_primary_metric(dose_4d, regime, mesh)
    penalty = m_4d - m_static

    pass_no = float(np.mean(m_no <= regime.constraint_limit))
    pass_static = float(np.mean(m_static <= regime.constraint_limit))
    n_pass_4d = int(np.sum(m_4d <= regime.constraint_limit))
    pass_4d = n_pass_4d / n_patients
    ci95 = wilson_ci(n_pass_4d, n_patients)

    # EBRT-only LKB NTCP (DESIGN.md §3: never transfer LKB to LDR)
    ntcp_no = ntcp_4d = None
    ntcp_rr: float | None = None
    if regime.modality == "ebrt":
        ntcp_no = lkb_ntcp(eqd2(dose_no, regime.n_fractions))
        ntcp_4d = lkb_ntcp(eqd2(dose_4d, regime.n_fractions))
        mean_no = float(np.mean(ntcp_no))
        ntcp_rr = float(np.mean(ntcp_4d)) / mean_no if mean_no > 1e-9 else None

    # Spearman variance attribution
    kin_driver = kin.early_drop_frac if material == "HA" else (
        kin.sep_half_life_days if material == "DuraSeal" else kin.onset_days
    )
    sensitivity = _spearman_shares(
        m_4d,
        {
            "mid_separation_mm": profiles.mid_mm,
            "apex_separation_mm": profiles.apex_mm,
            "resorption_kinetics": kin_driver,
            "oedema_contraction_mm": oedema_amp,
            "baseline_gap_mm": base_gap,
        },
    )

    if pass_4d >= 0.95:
        flag = "GREEN"
        rec = "High confidence (>=95%) of meeting rectal constraint after 4D resorption and motion."
    elif pass_4d >= 0.80:
        flag = "AMBER"
        rec = (
            "Borderline 4D margin (80-95% pass): static TPS overestimates protection; "
            "review posterior margin, adaptive verification imaging, or mid-course MRI."
        )
    else:
        flag = "RED"
        rec = (
            "Insufficient 4D rectal sparing (<80% pass after resorption/decay/motion): "
            "consider adaptive replanning, dose de-escalation, or faster-decay isotope / PEG switch."
        )

    return Simulation4DResult(
        regime=regime,
        material=material,
        n_patients=n_patients,
        seed=seed,
        metric_name=regime.constraint_metric,
        metric_unit=regime.constraint_unit,
        constraint_limit=regime.constraint_limit,
        metric_no_spacer=m_no,
        metric_static_tps=m_static,
        metric_4d_delivered=m_4d,
        resorption_penalty=penalty,
        protected_fraction=prot,
        pass_prob_no_spacer=pass_no,
        pass_prob_static_tps=pass_static,
        pass_prob_4d=pass_4d,
        pass_ci95_4d=ci95,
        ntcp_no_spacer=ntcp_no,
        ntcp_4d=ntcp_4d,
        ntcp_risk_ratio=ntcp_rr,
        sensitivity=sensitivity,
        risk_flag=flag,
        recommendation=rec,
    )


@dataclass(frozen=True)
class RequiredSeparationResult:
    """Inverse solution: minimum post-injection separation for target confidence (static vs 4D)."""

    regime_id: str
    material: str
    confidence: float
    grid_mm: list[float]
    pass_prob_static: list[float]
    pass_prob_4d: list[float]
    required_mm_static: float | None
    required_mm_4d: float | None
    hidden_margin_gap_mm: float | None  # required_4d - required_static

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_id": self.regime_id,
            "material": self.material,
            "confidence": self.confidence,
            "required_mm_static_tps": self.required_mm_static,
            "required_mm_4d_delivered": self.required_mm_4d,
            "hidden_margin_gap_mm": self.hidden_margin_gap_mm,
            "curve": [
                {"separation_mm": s, "pass_prob_static": round(ps, 4), "pass_prob_4d": round(p4, 4)}
                for s, ps, p4 in zip(self.grid_mm, self.pass_prob_static, self.pass_prob_4d, strict=True)
            ],
        }


def _first_crossing(xs: list[float], ys: list[float], target: float) -> float | None:
    for i, y in enumerate(ys):
        if y >= target:
            if i == 0:
                return round(xs[0], 2)
            y0, y1 = ys[i - 1], y
            x0, x1 = xs[i - 1], xs[i]
            if y1 <= y0:
                return round(x1, 2)
            return round(x0 + (target - y0) * (x1 - x0) / (y1 - y0), 2)
    return None


def solve_required_separation(
    regime_id: str,
    material: str = "PEG",
    *,
    confidence: float = 0.95,
    n_patients: int = 400,
    seed: int = 42,
    grid_mm: tuple[float, ...] = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0),
) -> RequiredSeparationResult:
    """Solve for the minimum achieved mid-gland separation achieving `confidence` constraint pass rate.

    Compares what a frozen-geometry commercial TPS thinks is enough (`required_mm_static`)
    against what is actually required once 4D resorption + isotope decay + oedema + motion act
    (`required_mm_4d`).
    """
    mesh = make_rectal_mesh()
    ps_list: list[float] = []
    p4_list: list[float] = []
    for s in grid_mm:
        res = simulate_4d(
            regime_id,
            material,
            n_patients=n_patients,
            seed=seed,
            mid_override_mm=s,
            include_placement_failures=False,
            mesh=mesh,
        )
        ps_list.append(res.pass_prob_static_tps)
        p4_list.append(res.pass_prob_4d)

    xs = [float(x) for x in grid_mm]
    req_static = _first_crossing(xs, ps_list, confidence)
    req_4d = _first_crossing(xs, p4_list, confidence)
    gap = round(req_4d - req_static, 2) if (req_4d is not None and req_static is not None) else None
    return RequiredSeparationResult(
        regime_id=regime_id,
        material=material,
        confidence=confidence,
        grid_mm=xs,
        pass_prob_static=ps_list,
        pass_prob_4d=p4_list,
        required_mm_static=req_static,
        required_mm_4d=req_4d,
        hidden_margin_gap_mm=gap,
    )


def compare_clock_matrix(
    regimes: tuple[str, ...] = ("ebrt_conv", "ebrt_sbrt", "ldr_i125", "ldr_pd103", "ldr_cs131", "ldr_i125_ebrt_boost", "hdr_ir192"),
    materials: tuple[str, ...] = ("PEG", "HA", "DuraSeal"),
    *,
    n_patients: int = 500,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Compare hidden 4D resorption penalties and protected dose shares across regimes and materials."""
    mesh = make_rectal_mesh()
    rows: list[dict[str, Any]] = []
    for reg in regimes:
        for mat in materials:
            r = simulate_4d(reg, mat, n_patients=n_patients, seed=seed, mesh=mesh)
            rows.append(
                {
                    "regime": reg,
                    "material": mat,
                    "metric": r.metric_name,
                    "unit": r.metric_unit,
                    "limit": r.constraint_limit,
                    "static_mean": round(float(np.mean(r.metric_static_tps)), 2),
                    "delivered_4d_mean": round(float(np.mean(r.metric_4d_delivered)), 2),
                    "resorption_penalty_mean": round(float(np.mean(r.resorption_penalty)), 2),
                    "protected_dose_pct": round(100.0 * float(np.mean(r.protected_fraction)), 1),
                    "pass_static_pct": round(100.0 * r.pass_prob_static_tps, 1),
                    "pass_4d_pct": round(100.0 * r.pass_prob_4d, 1),
                    "risk_flag": r.risk_flag,
                }
            )
    return rows
