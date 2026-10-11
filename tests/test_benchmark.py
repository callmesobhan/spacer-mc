"""M0 validation ladder, rung 1: the Monte Carlo must reproduce closed-form results."""

import numpy as np
import pytest

from spacer_mc import Scenario, falloff, simulate


def test_zero_uncertainty_matches_deterministic_profile():
    sc = Scenario(spacer_sd_mm=0.0, sigma_sys_mm=0.0, sigma_rand_mm=0.0, n_fractions=5)
    without, with_ = simulate(sc, n_patients=4, seed=0)
    assert without == pytest.approx(falloff(sc.gap_mm, sc.sigma_p_mm))
    assert with_ == pytest.approx(falloff(sc.gap_mm + sc.spacer_mean_mm, sc.sigma_p_mm))


def test_random_motion_blurs_penumbra_in_quadrature():
    # E over delta ~ N(0, s_r) of 0.5*erfc((x+delta)/(sqrt(2) s_p)) is the same profile
    # with sigma = sqrt(s_p^2 + s_r^2). This is the exact benchmark for the sampler.
    sc = Scenario(spacer_sd_mm=0.0, sigma_sys_mm=0.0, n_fractions=20)
    n = 20_000
    without, with_ = simulate(sc, n_patients=n, seed=1)
    sigma_tot = np.hypot(sc.sigma_p_mm, sc.sigma_rand_mm)
    for sample, x in ((without, sc.gap_mm), (with_, sc.gap_mm + sc.spacer_mean_mm)):
        expected = falloff(x, sigma_tot)
        stderr = sample.std(ddof=1) / np.sqrt(n)
        assert abs(sample.mean() - expected) < 4 * stderr


def test_common_random_numbers_make_spacer_never_worse():
    without, with_ = simulate(Scenario(), n_patients=5_000, seed=2)
    assert np.all(with_ <= without)


def test_seed_reproducibility():
    a = simulate(Scenario(), n_patients=100, seed=7)
    b = simulate(Scenario(), n_patients=100, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
