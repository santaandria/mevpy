"""Interface for the ordinary-event distribution F(x; theta) used inside the MEVD.

An ordinary distribution is a small object with four required members and two
optional ones.  The MEVD never needs anything else.

Required
--------
fit(sample) -> sequence of float
    Estimate theta from the events of ONE block.  ``sample`` is a 1-D float
    ndarray of finite values (length >= ``min_events``).  Return the
    parameters as a flat sequence of length ``len(param_names)``.  Raise any
    exception if the estimate cannot be computed; the MEVD reports it as a
    ``FitError`` naming the block.  Non-finite returned values are also
    rejected.
cdf(x, *theta) -> ndarray
    F(x; theta), continuous and strictly increasing between the support
    endpoints (required for a well-defined quantile inversion; flat stretches
    of the mixture then occur only in gaps between block supports).  Must
    follow numpy broadcasting: the MEVD calls it with
    ``x`` of shape ``(..., 1)`` and every parameter as an array of shape
    ``(m,)`` (one entry per block) and expects an array of shape ``(..., m)``.
    Write it with numpy ufuncs (as scipy.stats does) and this holds
    automatically.
ppf(p, *theta) -> ndarray
    Inverse of ``cdf`` with the same broadcasting rule.  Used only to build
    the starting bracket for the numerical inversion of the mixture.
support(*theta) -> (lower, upper)
    Support endpoints (arrays broadcast like the parameters; +-inf allowed).
    They describe the distribution's support, not the observed sample range.
    ``ppf(0)`` of the MEVD returns the smallest lower endpoint, ``ppf(1)`` the
    largest upper endpoint, and quantiles falling in a gap between block
    supports are moved to the left end of the gap (generalised inverse).

Optional (defaults are provided by the base class)
--------------------------------------------------
logcdf(x, *theta)  default ``log(cdf(...))``.  Override with an accurate
    formula when the upper tail matters: 1 - F^n is computed as
    ``-expm1(n * logcdf)``, so the accuracy of large return levels depends on
    it.
isf(s, *theta)     default ``ppf(1 - s)``.  Override for accuracy when s is tiny.

The MEVD stores a deep copy of the distribution object; do not modify a
distribution in place after it has been used to build a model.

Attributes
----------
param_names : tuple of str   names of theta, in the order returned by ``fit``.
min_events  : int            blocks with 0 < n < min_events cannot be fitted.
name        : str            used in messages and reprs.

Subclassing :class:`OrdinaryDistribution` is optional (duck typing works),
but it gives the defaults above.
"""
from __future__ import annotations

import numpy as np


class OrdinaryDistribution:
    """Base class for ordinary-event distributions (see module docstring)."""

    name = "ordinary"
    param_names: tuple = ()
    min_events = 1

    def fit(self, sample):
        raise NotImplementedError

    def cdf(self, x, *theta):
        raise NotImplementedError

    def ppf(self, p, *theta):
        raise NotImplementedError

    def support(self, *theta):
        raise NotImplementedError

    def logcdf(self, x, *theta):
        with np.errstate(divide="ignore"):
            return np.log(self.cdf(x, *theta))

    def isf(self, s, *theta):
        return self.ppf(1.0 - np.asarray(s, dtype=float), *theta)

    def __repr__(self):
        return f"{type(self).__name__}()"


class ScipyOrdinary(OrdinaryDistribution):
    """Wrap a ``scipy.stats`` continuous distribution as an ordinary distribution.

    theta is scipy's positional parameter tuple ``(*shapes, loc, scale)``.
    Fitting uses ``dist.fit(sample, **fit_kwargs)`` (scipy's maximum
    likelihood unless you pass ``method=...``); e.g.
    ``ScipyOrdinary(scipy.stats.gamma, floc=0)`` fixes loc at zero (loc is
    still part of theta).
    """

    def __init__(self, dist, min_events=2, **fit_kwargs):
        self.dist = dist
        self.fit_kwargs = fit_kwargs
        self.min_events = min_events
        shapes = dist.shapes.split(", ") if dist.shapes else []
        self.param_names = tuple(shapes) + ("loc", "scale")
        self.name = dist.name

    def fit(self, sample):
        return tuple(float(v) for v in self.dist.fit(sample, **self.fit_kwargs))

    def cdf(self, x, *theta):
        return self.dist.cdf(x, *theta)

    def logcdf(self, x, *theta):
        return self.dist.logcdf(x, *theta)

    def ppf(self, p, *theta):
        return self.dist.ppf(p, *theta)

    def isf(self, s, *theta):
        return self.dist.isf(s, *theta)

    def support(self, *theta):
        return self.dist.support(*theta)

    def __repr__(self):
        kw = "".join(f", {k}={v!r}" for k, v in self.fit_kwargs.items())
        return f"ScipyOrdinary({self.dist.name}{kw})"
