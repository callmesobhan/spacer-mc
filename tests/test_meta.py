"""The pooling code must reproduce a published meta-analysis from its own forest-plot values."""

import pathlib

import pytest
import yaml

from spacer_mc import dersimonian_laird

ROOT = pathlib.Path(__file__).resolve().parents[1]
PRIORS = yaml.safe_load((ROOT / "data" / "priors.yaml").read_text())
FOREST = PRIORS["ebrt_between_study_effects"]["rows"][0]["data"]


def _pool(exclude=()):
    rows = [r for r in FOREST if r["study"] not in exclude]
    return dersimonian_laird([r["md"] for r in rows], [r["se"] for r in rows])


def test_reproduces_miller_2020_published_result():
    # Miller 2020, JAMA Netw Open, Figure 1: pooled -6.51 (-10.51 to -2.51), tau^2 22.37, Q 159.40, I^2 97%
    pooled = _pool()
    assert pooled.mean == pytest.approx(-6.51, abs=0.01)
    assert pooled.ci95 == pytest.approx((-10.51, -2.51), abs=0.01)
    assert pooled.tau2 == pytest.approx(22.37, abs=0.01)
    assert pooled.q == pytest.approx(159.40, abs=0.05)
    assert pooled.i2 == pytest.approx(97, abs=0.5)


def test_weights_match_the_published_figure():
    # Figure 1 weights, in the order Chao, Mariados, Pinkawa, te Velde, Whalley, Wolf
    assert list(_pool().weights) == pytest.approx([18.6, 18.4, 17.5, 17.4, 13.6, 14.5], abs=0.1)


def test_without_chao_heterogeneity_collapses():
    # our own re-analysis: the HDR-metric study accounts for nearly all of the 97%
    pooled = _pool(exclude=("Chao 2019",))
    assert pooled.mean == pytest.approx(-7.90, abs=0.01)
    assert pooled.ci95 == pytest.approx((-9.58, -6.23), abs=0.01)
    assert pooled.tau2 == pytest.approx(1.63, abs=0.01)
    assert pooled.i2 == pytest.approx(50, abs=1)


def test_identical_studies_have_no_heterogeneity():
    pooled = dersimonian_laird([-5.0, -5.0, -5.0], [1.0, 1.0, 1.0])
    assert pooled.tau2 == 0.0 and pooled.i2 == 0.0 and pooled.mean == pytest.approx(-5.0)
