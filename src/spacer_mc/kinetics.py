"""4D hydrogel resorption kinetics, LDR prostate oedema decay, and brachytherapy isotope clocks.

Every kinetic and decay parameter is anchored to published sources catalogued in data/priors.yaml:
  - Isotope half-lives & prescriptions: King 2021 (Table 1), Henry 2022.
  - PEG (SpaceOAR-class) 3-phase kinetics: Mariados 2015, Pinkawa 2013, Uhl 2014, Kubo 2021.
  - Hyaluronic Acid (HA, Barrigel/Macrolane-class) fast early drop + plateau: Bjoreland 2023.
  - DuraSeal rapid resorption (separation half-life vs gel volume): Vaggers 2021 / Heikkila 2014.
  - Post-implant LDR prostate oedema decay (day 1 -> day 30 dose rise): Haddad 2023.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IsotopeSpec:
    """Published permanent/temporary brachytherapy isotope physics (King 2021 Table 1)."""

    name: str
    half_life_days: float
    mean_energy_kev: float
    mono_rx_gy: float
    boost_rx_gy: float
    source_bib: str = "King2021"

    def delivered_fraction(self, t_days: float | np.ndarray) -> np.ndarray:
        """Cumulative fraction of total decay dose delivered by time t_days."""
        t = np.maximum(np.asarray(t_days, dtype=float), 0.0)
        return 1.0 - np.power(2.0, -t / self.half_life_days)


ISOTOPES: dict[str, IsotopeSpec] = {
    "I-125": IsotopeSpec("I-125", half_life_days=59.4, mean_energy_kev=28.4, mono_rx_gy=145.0, boost_rx_gy=110.0),
    "Pd-103": IsotopeSpec("Pd-103", half_life_days=17.0, mean_energy_kev=20.7, mono_rx_gy=125.0, boost_rx_gy=100.0),
    "Cs-131": IsotopeSpec("Cs-131", half_life_days=9.7, mean_energy_kev=30.4, mono_rx_gy=115.0, boost_rx_gy=85.0),
    "Ir-192": IsotopeSpec("Ir-192", half_life_days=73.83, mean_energy_kev=380.0, mono_rx_gy=27.0, boost_rx_gy=15.0),
}

MATERIALS = ("PEG", "HA", "DuraSeal", "Balloon", "none")


@dataclass(frozen=True)
class KineticsDraw:
    """Per-patient sampled hydrogel separation kinetics parameters (shape (n_patients,))."""

    material: str
    onset_days: np.ndarray  # midpoint of steep hydrolysis/deflation (PEG/Balloon)
    transition_days: np.ndarray  # logistic transition scale tau (days)
    early_drop_frac: np.ndarray  # HA early 1-month drop fraction (~0.28-0.32)
    early_tau_days: np.ndarray  # HA early drop time constant (days)
    sep_half_life_days: np.ndarray  # exponential separation half-life (DuraSeal)


def sample_kinetics_params(
    material: str,
    n_patients: int,
    rng: np.random.Generator,
    *,
    deterministic: bool = False,
) -> KineticsDraw:
    """Sample patient-level hydrogel separation retention parameters.

    Models *rectal separation* rather than gel volume (Vaggers 2021 / Heikkila 2014).
    """
    if material not in MATERIALS:
        raise ValueError(f"Unknown spacer material {material!r}; choose from {MATERIALS}")

    def _draw(mean: float, sd: float, lo: float, hi: float) -> np.ndarray:
        if deterministic or sd <= 0.0:
            return np.full(n_patients, mean, dtype=float)
        return np.clip(rng.normal(mean, sd, size=n_patients), lo, hi)

    if material == "PEG":
        # Mariados 2015: 12.6 mm baseline -> 10.9 mm wk before mo 3 -> 6.8 mm wk after mo 3;
        # Uhl 2014: ~10.5 mm at 8 wk, ~2.9 mm at ~5 mo (~150 d), gone by 9-12 mo.
        # Kubo 2021: wide patient-to-patient onset variability around days 74-115.
        onset = _draw(96.0, 14.0, 60.0, 145.0)
        tau = _draw(18.0, 4.0, 8.0, 35.0)
        return KineticsDraw(
            material=material,
            onset_days=onset,
            transition_days=tau,
            early_drop_frac=np.zeros(n_patients),
            early_tau_days=np.ones(n_patients),
            sep_half_life_days=np.full(n_patients, np.inf),
        )

    if material == "HA":
        # Bjoreland 2023: HA thickness drops 28-32% within the first month (MR1 -> MR2 ~30 d),
        # then plateaus through end of EBRT (MR3 ~63 d), followed by slower late clearance (~6 mo).
        early_drop = _draw(0.30, 0.06, 0.12, 0.50)
        early_tau = _draw(14.0, 3.5, 6.0, 28.0)
        late_onset = _draw(180.0, 25.0, 120.0, 260.0)
        late_tau = _draw(30.0, 6.0, 15.0, 50.0)
        return KineticsDraw(
            material=material,
            onset_days=late_onset,
            transition_days=late_tau,
            early_drop_frac=early_drop,
            early_tau_days=early_tau,
            sep_half_life_days=np.full(n_patients, np.inf),
        )

    if material == "DuraSeal":
        # Vaggers 2021 quoting Heikkila 2014: gel volume T1/2 ~47 d, rectal separation T1/2 ~110 d
        # (and rapid breakdown reported at 4-6 wk in EBRT settings).
        t_half = _draw(65.0, 18.0, 25.0, 130.0)
        return KineticsDraw(
            material=material,
            onset_days=np.zeros(n_patients),
            transition_days=np.ones(n_patients),
            early_drop_frac=np.zeros(n_patients),
            early_tau_days=np.ones(n_patients),
            sep_half_life_days=t_half,
        )

    if material == "Balloon":
        # Biodegradable balloon: stable ~90-120 d, then rapid deflation over ~10-15 d.
        onset = _draw(110.0, 16.0, 70.0, 160.0)
        tau = _draw(10.0, 3.0, 4.0, 20.0)
        return KineticsDraw(
            material=material,
            onset_days=onset,
            transition_days=tau,
            early_drop_frac=np.zeros(n_patients),
            early_tau_days=np.ones(n_patients),
            sep_half_life_days=np.full(n_patients, np.inf),
        )

    # material == "none"
    return KineticsDraw(
        material="none",
        onset_days=np.zeros(n_patients),
        transition_days=np.ones(n_patients),
        early_drop_frac=np.zeros(n_patients),
        early_tau_days=np.ones(n_patients),
        sep_half_life_days=np.ones(n_patients),
    )


def spacer_retention(t_days: np.ndarray, draw: KineticsDraw) -> np.ndarray:
    """Return separation retention fraction r(t) in [0, 1] of shape (n_patients, n_times).

    Normalized so r(0) = 1.0 for every active spacer material (0.0 for 'none').
    """
    t = np.maximum(np.asarray(t_days, dtype=float), 0.0)[None, :]  # (1, T)
    n_p = len(draw.onset_days)
    if draw.material == "none":
        return np.zeros((n_p, t.shape[1]), dtype=float)

    if draw.material in ("PEG", "Balloon"):
        # Logistic drop normalized to 1.0 at t=0:
        # s(t) = (1 + exp(-onset/tau)) / (1 + exp((t - onset)/tau))
        onset = draw.onset_days[:, None]
        tau = draw.transition_days[:, None]
        num = 1.0 + np.exp(-onset / tau)
        den = 1.0 + np.exp(np.clip((t - onset) / tau, -60.0, 60.0))
        return np.clip(num / den, 0.0, 1.0)

    if draw.material == "HA":
        # Fast early drop of early_drop_frac over ~1 month, plateau, then late biodegradation.
        f_drop = draw.early_drop_frac[:, None]
        tau_e = draw.early_tau_days[:, None]
        early_factor = (1.0 - f_drop) + f_drop * np.exp(-t / tau_e)

        onset = draw.onset_days[:, None]
        tau_l = draw.transition_days[:, None]
        late_factor = (1.0 + np.exp(-onset / tau_l)) / (
            1.0 + np.exp(np.clip((t - onset) / tau_l, -60.0, 60.0))
        )
        return np.clip(early_factor * late_factor, 0.0, 1.0)

    if draw.material == "DuraSeal":
        t_half = draw.sep_half_life_days[:, None]
        return np.power(2.0, -t / t_half)

    raise ValueError(f"Unhandled spacer material {draw.material!r}")


def oedema_contraction_mm(
    t_days: np.ndarray,
    delta_max_mm: np.ndarray,
    t_half_days: float = 10.0,
) -> np.ndarray:
    """LDR post-implant prostate oedema resolution shifting sources closer to rectum over month 1.

    At t=0 (implant day), prostate is maximally oedematous (contraction = 0).
    As oedema resolves with half-life ~10 d, effective source-to-rectum distance shrinks by
    up to delta_max_mm (reproducing Haddad 2023 day 1 -> day 30 rectal D2cc rise of ~14% relative).
    Returns positive contraction in mm of shape (n_patients, n_times) to SUBTRACT from gap.
    """
    t = np.maximum(np.asarray(t_days, dtype=float), 0.0)[None, :]
    return delta_max_mm[:, None] * (1.0 - np.power(2.0, -t / t_half_days))


def decay_quadrature(half_life_days: float, n_steps: int = 80, n_half_lives: float = 6.5) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint quadrature (t_days, weights) for radioactive decay integrating to ~99% of infinity.

    Weights are exact analytical integrals of lambda * exp(-lambda * t) over each bin,
    normalized to sum to 1.0 (representing 100% of infinite-time delivered dose).
    """
    t_max = half_life_days * n_half_lives
    # Log-stretched grid resolves fast early oedema/HA changes AND long I-125 tail accurately.
    edges = np.geomspace(0.25, t_max + 0.25, n_steps + 1) - 0.25
    edges[0] = 0.0
    mids = 0.5 * (edges[:-1] + edges[1:])
    lam = np.log(2.0) / half_life_days
    raw_w = np.exp(-lam * edges[:-1]) - np.exp(-lam * edges[1:])
    weights = raw_w / raw_w.sum()
    return mids, weights


def fraction_schedule_days(n_fractions: int, start_day: float = 7.0, fx_per_week: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Treatment-day timestamps and equal per-fraction weights for EBRT / HDR schedules."""
    if n_fractions <= 0:
        raise ValueError("n_fractions must be positive")
    if n_fractions == 1:
        return np.array([start_day], dtype=float), np.array([1.0], dtype=float)
    step = 7.0 / fx_per_week
    days = start_day + np.arange(n_fractions, dtype=float) * step
    weights = np.full(n_fractions, 1.0 / n_fractions, dtype=float)
    return days, weights


def protected_dose_fraction(
    t_days: np.ndarray,
    time_weights: np.ndarray,
    initial_mid_mm: np.ndarray,
    draw: KineticsDraw,
    *,
    threshold_mm: float = 5.0,
) -> np.ndarray:
    """Fraction of total prescribed dose delivered while mid-gland spacer separation >= threshold_mm."""
    sep_t = initial_mid_mm[:, None] * spacer_retention(t_days, draw)
    is_protected = (sep_t >= threshold_mm).astype(float)
    return is_protected @ time_weights
