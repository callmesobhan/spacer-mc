"""Dose engines. M0 ships only the analytic 1-D falloff; see DESIGN.md for the interface."""

import numpy as np
from scipy.special import erfc


def falloff(s_mm, sigma_p_mm, s50_mm=0.0):
    """Fraction of prescription dose at distance s_mm along the anterior-posterior axis.

    The profile is a Gaussian-blurred step: 0.5 at s50_mm, with penumbra width sigma_p_mm.
    """
    return 0.5 * erfc((np.asarray(s_mm) - s50_mm) / (np.sqrt(2.0) * sigma_p_mm))
