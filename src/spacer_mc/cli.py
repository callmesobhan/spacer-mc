"""Command-line interface for spacer-mc 4D hydrogel resorption x isotope-decay decision engine."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

import numpy as np

from .decision import (
    REGIMES,
    compare_clock_matrix,
    simulate_4d,
    solve_required_separation,
)
from .dose import EBRTGradientEngine, TG43ProstateEngine
from .geometry import make_rectal_mesh, sample_daily_motion_mm
from .kinetics import (
    ISOTOPES,
    MATERIALS,
    oedema_contraction_mm,
    sample_kinetics_params,
    spacer_retention,
)
from .radbio import d_xcc, v_dose_pct


def _cmd_simulate(args: argparse.Namespace) -> int:
    profile = None
    if args.mid is not None:
        apex = args.apex if args.apex is not None else round(args.mid * 0.82, 2)
        base = args.base if args.base is not None else round(args.mid * 1.12, 2)
        profile = (apex, args.mid, base)
    res = simulate_4d(
        args.regime,
        args.material,
        n_patients=args.n,
        seed=args.seed,
        patient_profile_mm=profile,
        asymmetry_mm=args.asymmetry,
    )
    payload = res.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"spacer-mc 4D Simulation — {res.regime.label}")
    print(f"  Spacer material:  {res.material}" + (f"  (patient apex/mid/base = {profile} mm)" if profile else "  (population cohort)"))
    print(f"  Constraint:       {res.metric_name} <= {res.constraint_limit} {res.metric_unit}  [{res.regime.constraint_source}]")
    print()
    print(f"  No spacer:        {payload['no_spacer']['mean']:.2f} +/- {payload['no_spacer']['sd']:.2f} {res.metric_unit}  (pass {payload['pass_probability']['no_spacer']*100:.1f}%)")
    print(f"  Static TPS (d0):  {payload['static_tps']['mean']:.2f} +/- {payload['static_tps']['sd']:.2f} {res.metric_unit}  (pass {payload['pass_probability']['static_tps']*100:.1f}%)")
    print(f"  4D Delivered:     {payload['delivered_4d']['mean']:.2f} +/- {payload['delivered_4d']['sd']:.2f} {res.metric_unit}  (pass {payload['pass_probability']['delivered_4d']*100:.1f}%, 95% CI {payload['pass_probability']['delivered_4d_ci95'][0]*100:.1f}-{payload['pass_probability']['delivered_4d_ci95'][1]*100:.1f}%)")
    print(f"  Hidden penalty:   +{payload['hidden_resorption_penalty']['mean']:.2f} {res.metric_unit} (4D delivered minus static TPS)")
    print(f"  Dose protected:   {payload['protected_dose_fraction']['mean']*100:.1f}% delivered while separation >= 5 mm")
    if payload["ntcp"] is not None:
        n = payload["ntcp"]
        print(f"  LKB NTCP (>=G2):  {n['no_spacer_mean_pct']:.2f}% (no spacer) -> {n['delivered_4d_mean_pct']:.2f}% (4D spacer), RR = {n['risk_ratio']}")
    print(f"\n  Decision flag:    [{res.risk_flag}] {res.recommendation}")
    return 0


def _cmd_compare_clocks(args: argparse.Namespace) -> int:
    rows = compare_clock_matrix(n_patients=args.n, seed=args.seed)
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    header = (
        f"{'Regime':20} {'Mat':8} {'Metric':12} {'Static':>7} {'4D Deliv':>8} "
        f"{'Penalty':>8} {'Prot%':>6} {'Pass(S)':>8} {'Pass(4D)':>9} {'Flag':6}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['regime']:20} {r['material']:8} {r['metric']:12} "
            f"{r['static_mean']:7.2f} {r['delivered_4d_mean']:8.2f} "
            f"{r['resorption_penalty_mean']:+8.2f} {r['protected_dose_pct']:6.1f} "
            f"{r['pass_static_pct']:7.1f}% {r['pass_4d_pct']:8.1f}% {r['risk_flag']:6}"
        )
    return 0


def _cmd_required_separation(args: argparse.Namespace) -> int:
    sol = solve_required_separation(
        args.regime,
        args.material,
        confidence=args.confidence,
        n_patients=args.n,
        seed=args.seed,
    )
    payload = sol.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    print(f"Inverse Required Separation Map — {args.regime} / {args.material} (target confidence {args.confidence*100:.0f}%)")
    print(f"  Static frozen-geometry TPS requires:  {sol.required_mm_static} mm")
    print(f"  4D resorption+decay+motion requires:  {sol.required_mm_4d} mm")
    print(f"  Hidden required-separation gap:       {sol.hidden_margin_gap_mm} mm")
    print()
    print(f"  {'Sep (mm)':>8} {'Pass Static':>12} {'Pass 4D':>12}")
    for row in payload["curve"]:
        print(f"  {row['separation_mm']:8.1f} {row['pass_prob_static']*100:11.1f}% {row['pass_prob_4d']*100:11.1f}%")
    return 0


def _cmd_validate(_: argparse.Namespace) -> int:
    """Run literature benchmark verification against published sources in data/priors.yaml."""
    mesh = make_rectal_mesh()
    ebrt = EBRTGradientEngine()
    susil = {s: float(v_dose_pct(ebrt.dose_fraction(mesh, np.full((1, mesh.n_voxels), s)) * 78.0, 70.0)[0]) for s in (0.0, 5.0, 10.0, 15.0)}

    rng = np.random.default_rng(0)
    peg_kin = sample_kinetics_params("PEG", 1, rng, deterministic=True)
    ha_kin = sample_kinetics_params("HA", 1, rng, deterministic=True)
    peg_t = spacer_retention(np.array([0.0, 56.0, 84.0, 98.0, 150.0]), peg_kin)[0]
    ha_t = spacer_retention(np.array([0.0, 30.0, 63.0]), ha_kin)[0]

    mot = sample_daily_motion_mm(4000, 10, np.random.default_rng(1), with_spacer=False, sys_sd_mm=0.0)
    # Remove the -0.5 mean drift to check pure Polizzi 2024 centered spread
    mot_c = mot - np.mean(mot)
    within5 = float(np.mean(np.abs(mot_c) <= 5.0))
    beyond10 = float(np.mean(np.abs(mot_c) > 10.0))

    tg43_i125 = TG43ProstateEngine("I-125")
    oed_d30 = oedema_contraction_mm(np.array([1.0, 30.0]), np.array([1.8]))[0]
    d2_d1 = float(d_xcc(tg43_i125.dose_fraction(mesh, np.full((1, mesh.n_voxels), 2.0 - oed_d30[0])) * 100.0, 2.0, mesh.voxel_vol_cc)[0])
    d2_d30 = float(d_xcc(tg43_i125.dose_fraction(mesh, np.full((1, mesh.n_voxels), 2.0 - oed_d30[1])) * 100.0, 2.0, mesh.voxel_vol_cc)[0])

    ebrt_res = simulate_4d("ebrt_conv", "PEG", n_patients=800, seed=42)

    print("spacer-mc Literature Benchmark Check")
    print(f"  1. King 2021 decay @ 90 d:     I-125={ISOTOPES['I-125'].delivered_fraction(90)*100:.1f}% (ref 65.0%), Pd-103={ISOTOPES['Pd-103'].delivered_fraction(90)*100:.1f}% (ref 97.4%), Cs-131={ISOTOPES['Cs-131'].delivered_fraction(90)*100:.1f}% (ref 99.8%)")
    print(f"  2. Susil 2010 pure-shift V70:  0mm={susil[0.0]:.1f}% (23.0%), 5mm={susil[5.0]:.1f}% (15.1%), 10mm={susil[10.0]:.1f}% (3.7%), 15mm={susil[15.0]:.1f}% (0.0%)")
    print(f"  3. Mariados/Uhl PEG retention: d56={peg_t[1]*100:.0f}%, d84={peg_t[2]*100:.0f}%, d98={peg_t[3]*100:.0f}%, d150={peg_t[4]*100:.0f}%")
    print(f"  4. Bjoreland 2023 HA drop:     d30={ha_t[1]*100:.0f}% (-{(1-ha_t[1])*100:.0f}%, ref -28%), d63={ha_t[2]*100:.0f}% (-{(1-ha_t[2])*100:.0f}%, ref -32%)")
    print(f"  5. Polizzi 2024 daily motion:  |d|<=5mm={within5*100:.1f}% (ref 77.2%), |d|>10mm={beyond10*100:.1f}% (ref 3.6%)")
    print(f"  6. Haddad 2023 LDR oedema:     no-spacer D2cc day1={d2_d1:.1f}% -> day30={d2_d30:.1f}% (+{(d2_d30/d2_d1 - 1)*100:.1f}% rel, ref +14%)")
    print(f"  7. Wong 2025 late G>=2 NTCP:   RR={ebrt_res.ntcp_risk_ratio:.2f} (ref 0.25, 95% CI 0.15-0.42)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spacer-mc",
        description="4D coupled hydrogel resorption x isotope-decay x motion simulator & inverse separation solver.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="run 4D paired Monte Carlo simulation for a cohort or specific patient")
    s.add_argument("--regime", choices=tuple(REGIMES), required=True)
    s.add_argument("--material", choices=MATERIALS, default="PEG")
    s.add_argument("--mid", type=float, help="patient post-injection mid-gland separation (mm)")
    s.add_argument("--apex", type=float, help="patient post-injection apex separation (mm)")
    s.add_argument("--base", type=float, help="patient post-injection base separation (mm)")
    s.add_argument("--asymmetry", type=float, default=0.0, help="lateral offset asymmetry (mm)")
    s.add_argument("--n", type=int, default=1000, help="number of Monte Carlo draws")
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=_cmd_simulate)

    c = sub.add_parser("compare-clocks", help="compare 4D resorption penalties across isotopes, EBRT, and materials")
    c.add_argument("--n", type=int, default=500)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=_cmd_compare_clocks)

    r = sub.add_parser("required-separation", help="solve inverse minimum separation for target confidence (static vs 4D)")
    r.add_argument("--regime", choices=tuple(REGIMES), required=True)
    r.add_argument("--material", choices=MATERIALS, default="PEG")
    r.add_argument("--confidence", type=float, default=0.95)
    r.add_argument("--n", type=int, default=400)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--json", action="store_true")
    r.set_defaults(func=_cmd_required_separation)

    v = sub.add_parser("validate", help="check calibrated 4D engine against published benchmarks in data/priors.yaml")
    v.set_defaults(func=_cmd_validate)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
