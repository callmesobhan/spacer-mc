"""Physics, literature-benchmark, and decision-engine tests for 4D hydrogel x isotope-decay simulation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from spacer_mc import (
    ISOTOPES,
    EBRTGradientEngine,
    TG43ProstateEngine,
    compare_clock_matrix,
    d_xcc,
    make_rectal_mesh,
    sample_kinetics_params,
    simulate_4d,
    solve_required_separation,
    spacer_retention,
    v_dose_pct,
)
from spacer_mc.cli import main
from spacer_mc.geometry import sample_daily_motion_mm
from spacer_mc.kinetics import oedema_contraction_mm


def test_king2021_isotope_decay_clocks():
    """Exact radioactive decay shares by day 90 match King 2021 Table 1 / DESIGN.md §1."""
    assert ISOTOPES["I-125"].delivered_fraction(90.0) == pytest.approx(0.650, abs=0.005)
    assert ISOTOPES["Pd-103"].delivered_fraction(90.0) == pytest.approx(0.974, abs=0.005)
    assert ISOTOPES["Cs-131"].delivered_fraction(90.0) == pytest.approx(0.998, abs=0.002)


def test_peg_vs_ha_resorption_kinetics_match_mariados_uhl_and_bjoreland():
    rng = np.random.default_rng(0)
    peg = sample_kinetics_params("PEG", 1, rng, deterministic=True)
    ha = sample_kinetics_params("HA", 1, rng, deterministic=True)

    # PEG: stable through 8-week EBRT (day 56 > 85%), steep drop around 3-5 months (day 150 < 20%)
    r_peg = spacer_retention(np.array([0.0, 56.0, 96.0, 150.0, 300.0]), peg)[0]
    assert r_peg[0] == pytest.approx(1.0)
    assert r_peg[1] > 0.85
    assert 0.45 < r_peg[2] < 0.55
    assert r_peg[3] < 0.12
    assert r_peg[4] < 0.01

    # HA (Bjoreland 2023): loses ~25-32% in first month, then plateaus through end of EBRT (day 63)
    r_ha = spacer_retention(np.array([0.0, 30.0, 63.0]), ha)[0]
    assert r_ha[0] == pytest.approx(1.0)
    assert 0.65 < r_ha[1] < 0.78
    assert abs(r_ha[2] - r_ha[1]) < 0.08  # plateau behavior during weeks 4-9


def test_tg43_lut_matches_exact_kernel():
    mesh = make_rectal_mesh()
    eng = TG43ProstateEngine("I-125")
    sep = np.full((1, mesh.n_voxels), 8.3)
    approx = eng.dose_fraction(mesh, sep)[0]
    exact = eng._raw_kernel(mesh.x_mm, mesh.y_base_mm + 8.3, mesh.z_mm) / eng._norm
    assert np.max(np.abs(approx - exact) / exact) < 5e-4


def test_susil2010_pure_shift_ebrt_v70_saturation():
    mesh = make_rectal_mesh()
    ebrt = EBRTGradientEngine()
    v70 = [
        float(v_dose_pct(ebrt.dose_fraction(mesh, np.full((1, mesh.n_voxels), s)) * 78.0, 70.0)[0])
        for s in (0.0, 5.0, 10.0, 15.0)
    ]
    # Monotonic reduction, saturating near 0% by 15 mm (Susil 2010: 23.0 -> 15.1 -> 3.7 -> 0.0)
    assert v70[0] > v70[1] > v70[2] >= v70[3]
    assert 17.0 <= v70[0] <= 29.0
    assert v70[2] <= 7.0
    assert v70[3] <= 1.0


def test_polizzi2024_heavy_tailed_daily_motion():
    mot = sample_daily_motion_mm(5000, 10, np.random.default_rng(7), with_spacer=False, sys_sd_mm=0.0)
    centered = mot - np.mean(mot)
    within5 = float(np.mean(np.abs(centered) <= 5.0))
    beyond10 = float(np.mean(np.abs(centered) > 10.0))
    assert 0.72 <= within5 <= 0.82
    assert 0.025 <= beyond10 <= 0.055


def test_haddad2023_ldr_oedema_raises_day30_rectal_dose():
    mesh = make_rectal_mesh()
    tg43 = TG43ProstateEngine("I-125")
    oed = oedema_contraction_mm(np.array([1.0, 30.0]), np.array([1.8]))[0]
    d2_d1 = float(d_xcc(tg43.dose_fraction(mesh, np.full((1, mesh.n_voxels), 2.0 - oed[0])) * 100.0, 2.0, mesh.voxel_vol_cc)[0])
    d2_d30 = float(d_xcc(tg43.dose_fraction(mesh, np.full((1, mesh.n_voxels), 2.0 - oed[1])) * 100.0, 2.0, mesh.voxel_vol_cc)[0])
    rel_increase = (d2_d30 - d2_d1) / d2_d1
    assert 0.08 <= rel_increase <= 0.25


def test_core_4d_hypothesis_i125_suffers_larger_resorption_penalty_than_pd103_and_cs131():
    """The central thesis of DESIGN.md: PEG covers ~97-100% of Pd-103/Cs-131 but only ~65% of I-125."""
    r_i125 = simulate_4d("ldr_i125", "PEG", n_patients=400, seed=11, mid_override_mm=10.0, include_motion=False, include_oedema=False)
    r_pd103 = simulate_4d("ldr_pd103", "PEG", n_patients=400, seed=11, mid_override_mm=10.0, include_motion=False, include_oedema=False)
    r_cs131 = simulate_4d("ldr_cs131", "PEG", n_patients=400, seed=11, mid_override_mm=10.0, include_motion=False, include_oedema=False)

    assert np.mean(r_i125.protected_fraction) < 0.78
    assert np.mean(r_pd103.protected_fraction) > 0.94
    assert np.mean(r_cs131.protected_fraction) > 0.98
    assert np.mean(r_i125.resorption_penalty) > np.mean(r_pd103.resorption_penalty) > np.mean(r_cs131.resorption_penalty)


def test_ebrt_ha_suffers_early_resorption_penalty_and_ntcp_rr_matches_wong2025():
    r_peg = simulate_4d("ebrt_conv", "PEG", n_patients=500, seed=42)
    r_ha = simulate_4d("ebrt_conv", "HA", n_patients=500, seed=42)
    # Wong 2025: late Grade >=2 rectal toxicity RR = 0.25 (95% CI 0.15-0.42)
    assert r_peg.ntcp_risk_ratio is not None
    assert 0.12 <= r_peg.ntcp_risk_ratio <= 0.45
    # HA early month-1 loss yields a larger 4D penalty over 8-week EBRT than PEG
    assert np.mean(r_ha.resorption_penalty) > np.mean(r_peg.resorption_penalty)


def test_inverse_required_separation_exposes_4d_hidden_margin_gap():
    sol_i125 = solve_required_separation("ldr_i125", "PEG", confidence=0.95, n_patients=250, seed=5)
    assert sol_i125.required_mm_static is not None
    assert sol_i125.required_mm_4d is not None
    # True 4D delivery requires strictly more post-injection separation than static frozen TPS predicts
    assert sol_i125.required_mm_4d > sol_i125.required_mm_static
    assert sol_i125.hidden_margin_gap_mm is not None and sol_i125.hidden_margin_gap_mm > 0.0


def test_clock_matrix_and_cli_commands(capsys):
    rows = compare_clock_matrix(regimes=("ebrt_conv", "ldr_i125", "ldr_pd103"), materials=("PEG", "HA"), n_patients=100, seed=3)
    assert len(rows) == 6

    assert main(["simulate", "--regime", "ldr_i125", "--material", "PEG", "--mid", "10", "--n", "100", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["regime_id"] == "ldr_i125"
    assert payload["risk_flag"] in ("GREEN", "AMBER", "RED")

    assert main(["required-separation", "--regime", "ebrt_conv", "--material", "HA", "--n", "100", "--json"]) == 0
    inv = json.loads(capsys.readouterr().out)
    assert "required_mm_4d_delivered" in inv

    assert main(["validate"]) == 0
    val_out = capsys.readouterr().out
    assert "King 2021" in val_out and "Wong 2025" in val_out
