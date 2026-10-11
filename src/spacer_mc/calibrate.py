"""Random-effects pooling, used to calibrate and validate against published between-study results."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pooled:
    mean: float
    se: float
    tau2: float  # between-study variance
    q: float  # Cochran's Q
    i2: float  # percent of variation due to between-study heterogeneity
    weights: np.ndarray  # random-effects weights, percent

    @property
    def ci95(self) -> tuple[float, float]:
        return self.mean - 1.96 * self.se, self.mean + 1.96 * self.se


def dersimonian_laird(effects, ses) -> Pooled:
    """DerSimonian-Laird random-effects pooling of study effects with their standard errors."""
    y, v = np.asarray(effects, float), np.asarray(ses, float) ** 2
    w = 1.0 / v
    fixed = (w * y).sum() / w.sum()
    q = float((w * (y - fixed) ** 2).sum())
    k = len(y)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    ws = 1.0 / (v + tau2)
    mean = float((ws * y).sum() / ws.sum())
    i2 = max(0.0, (q - (k - 1)) / q) * 100.0 if q > 0 else 0.0
    return Pooled(mean, float((1.0 / ws.sum()) ** 0.5), float(tau2), q, i2, ws / ws.sum() * 100.0)
