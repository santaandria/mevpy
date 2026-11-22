"""Two-parameter Weibull fitted by probability-weighted moments (Greenwood 1979).

Adapted from mevpy v1 (E. Zorzetto; ``wei_fit_pwm``).

The ordinary-event model used for daily rainfall in the MEVD literature
(Marani & Ignaccolo 2015; Zorzetto et al. 2016): a Weibull distribution of the
daily depths above a threshold, fitted separately to each year.  The threshold
is the location ``loc``, so the model works directly in depth units:

    F(x; C, W) = 1 - exp(-((x - loc) / C) ** W),   x >= loc,

and ``fit`` applies the PWM estimator to the excesses ``sample - loc``.
Return levels therefore come out as depths.

PWM estimator: with the ordered excesses x_(1) <= ... <= x_(n),
M0 = mean(x) and M1 = sum_i x_(i) (n - i) / (n (n - 1)),

    W = ln 2 / ln(M0 / (2 M1)),     C = M0 / Gamma(1 + 1 / W).
"""
from __future__ import annotations

import numpy as np
from scipy.special import gamma

from .ordinary import OrdinaryDistribution

_LN2 = np.log(2.0)


class WeibullPWM(OrdinaryDistribution):
    """Weibull(scale C, shape W) above ``loc``, fitted by PWM.

    theta = (C, W).  Every event passed to ``fit`` must be strictly greater
    than ``loc`` (select them so, e.g. daily depth > threshold); values at or
    below ``loc`` raise instead of being silently dropped, because dropping
    them would make the block's event count n disagree with its fit.
    """

    name = "weibull_pwm"
    param_names = ("C", "W")
    min_events = 2                  # M1 divides by n (n - 1)

    def __init__(self, loc=0.0):
        self._loc = float(loc)

    @property
    def loc(self):
        """Location (threshold); read-only, because fitted parameters depend on it."""
        return self._loc

    def fit(self, sample):
        x = np.sort(np.asarray(sample, dtype=float) - self.loc)
        if x.size < 2:
            raise ValueError(f"PWM needs >= 2 events, got {x.size}")
        if not np.all(x > 0):
            raise ValueError(f"all events must be > loc={self.loc}")
        n = x.size
        m0 = x.mean()
        m1 = np.sum(x * (n - np.arange(1, n + 1))) / (n * (n - 1))
        with np.errstate(all="ignore"):
            c = m0 / gamma(np.log(m0 / m1) / _LN2)        # = M0 / Gamma(1 + 1/W)
            w = _LN2 / np.log(m0 / (2.0 * m1))
        if not (np.isfinite(c) and np.isfinite(w) and c > 0 and w > 0):
            raise ValueError(f"PWM gave invalid parameters C={c}, W={w} (e.g. all events equal)")
        return float(c), float(w)

    def _z(self, x, c, w):
        with np.errstate(invalid="ignore"):
            return (np.maximum(np.asarray(x, dtype=float) - self.loc, 0.0) / c) ** w

    def cdf(self, x, c, w):
        return -np.expm1(-self._z(x, c, w))

    def logcdf(self, x, c, w):
        # log(1 - exp(-z)) accurate in both tails ("log1mexp", Maechler 2012).
        z = self._z(x, c, w)
        with np.errstate(divide="ignore"):
            return np.where(z > _LN2, np.log1p(-np.exp(-z)), np.log(-np.expm1(-z)))

    def ppf(self, p, c, w):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.loc + c * (-np.log1p(-np.asarray(p, dtype=float))) ** (1.0 / w)

    def isf(self, s, c, w):
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.loc + c * (-np.log(np.asarray(s, dtype=float))) ** (1.0 / w)

    def support(self, c, w):
        shape = np.broadcast(np.asarray(c), np.asarray(w)).shape
        return np.full(shape, self.loc), np.full(shape, np.inf)

    def __repr__(self):
        return f"WeibullPWM(loc={self.loc:g})"
