"""Nonparametric block bootstrap of MEVD return levels.

Resampling unit
    Whole blocks (years), drawn with replacement, M draws for a model of M
    blocks.  A draw carries its block's (n_j, theta_j) pair; repeated draws
    are separate block instances (positions, never merged by label).
    Resampling whole blocks assumes
    blocks are independent and identically distributed, and keeps the
    within-block dependence (including that between n_j and the magnitudes).
Fitting
    The replicate MEVD is built from the drawn (n_j, theta_j).  Because each
    theta_j is estimated from block j's events only, this is identical to
    refitting the resampled blocks for any deterministic per-block estimator.
    A custom distribution therefore needs nothing extra; if its fit is
    stochastic, the replicates reuse the point-estimate fits.
Interval
    Percentile interval of the valid replicate estimates, per return period
    (``ci_level=0.90`` gives the 5th/95th percentiles).  It is
    reported only if at least ``min_valid_count`` replicates and a fraction
    ``min_valid_frac`` of all replicates are valid; otherwise NaN with a
    status saying why.
Failures
    A replicate return level whose inversion fails is NaN, its status holds
    the error message, and it is excluded only from that return period's
    interval.  Nothing is dropped silently: ``status``, ``n_valid`` and
    ``failures()`` report every exclusion.
Random state
    ``random_state`` is anything ``numpy.random.default_rng`` accepts (None,
    int, SeedSequence, Generator).  The draws are
    ``rng.integers(0, M, size=(n_boot, M))``; the indices are returned, so
    other statistics (e.g. a GEV fit) can be computed on the same draws.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .core import InversionError

OK = "ok"


def percentile_interval(values, valid, ci_level=0.90, min_valid_count=10, min_valid_frac=0.90):
    """Percentile interval from the valid entries of ``values``.

    Returns (lower, upper, status); status is "ok", "no_bootstrap",
    "insufficient_valid_count" or "insufficient_valid_fraction".
    """
    values = np.asarray(values, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    nb = values.size
    v = values[valid]
    if nb == 0:
        return np.nan, np.nan, "no_bootstrap"
    if v.size < min_valid_count:
        return np.nan, np.nan, "insufficient_valid_count"
    if v.size / nb < min_valid_frac:
        return np.nan, np.nan, "insufficient_valid_fraction"
    p = round(50 * (1 - ci_level), 10)          # 0.90 -> exactly [5, 95]
    lo, hi = np.percentile(v, [p, 100 - p])
    return float(lo), float(hi), OK


def return_levels_with_status(model, return_periods):
    """Return levels and per-level status ("ok" or the InversionError message)."""
    T = np.atleast_1d(np.asarray(return_periods, dtype=float))
    levels = np.full(T.shape, np.nan)
    status = np.full(T.shape, OK, dtype=object)
    for k, t in enumerate(T):
        try:
            levels[k] = model.isf(1.0 / t)
        except InversionError as exc:
            status[k] = f"inversion_error: {exc}"
    return levels, status


@dataclass
class BootstrapResult:
    return_periods: np.ndarray      # (nT,)
    estimate: np.ndarray            # (nT,) point estimates from the original blocks
    point_status: np.ndarray        # (nT,)
    lower: np.ndarray               # (nT,) interval bounds (NaN if withheld)
    upper: np.ndarray
    ci_status: np.ndarray           # (nT,)
    replicates: np.ndarray          # (n_boot, nT), NaN where failed
    status: np.ndarray              # (n_boot, nT), "ok" or error message
    indices: np.ndarray             # (n_boot, M) drawn block positions
    ci_level: float

    @property
    def valid(self):
        return self.status == OK

    @property
    def n_valid(self):
        return self.valid.sum(axis=0)

    def failures(self):
        """DataFrame with one row per failed (replicate, return period)."""
        import pandas as pd
        r, k = np.nonzero(~self.valid)
        return pd.DataFrame({"replicate": r, "T": self.return_periods[k],
                             "status": self.status[r, k]})

    def summary(self):
        """DataFrame with one row per return period."""
        import pandas as pd
        return pd.DataFrame({"estimate": self.estimate, "point_status": self.point_status,
                             "lower": self.lower, "upper": self.upper, "ci_status": self.ci_status,
                             "n_boot": self.replicates.shape[0], "n_valid": self.n_valid},
                            index=pd.Index(self.return_periods, name="T"))


def bootstrap_indices(n_blocks, n_boot, random_state=None):
    """Block positions (n_boot, n_blocks) drawn with replacement."""
    rng = np.random.default_rng(random_state)
    return rng.integers(0, n_blocks, size=(n_boot, n_blocks))


def bootstrap(model, return_periods, n_boot=1000, ci_level=0.90, random_state=None,
              min_valid_count=10, min_valid_frac=0.90):
    """Bootstrap percentile intervals of return levels (see module docstring).

    model          : fitted :class:`mevpy.MEVD` (any ordinary distribution).
    return_periods : scalar or sequence, T > 1 in block units.
    """
    T = np.atleast_1d(np.asarray(return_periods, dtype=float))
    if not np.all(np.isfinite(T) & (T > 1)):
        raise ValueError("return periods must be finite and > 1 (in block units)")
    if int(n_boot) != n_boot or n_boot < 0:
        raise ValueError("n_boot must be a non-negative integer")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be in (0, 1)")
    est, est_status = return_levels_with_status(model, T)
    idx = bootstrap_indices(model.n_blocks, int(n_boot), random_state)
    reps = np.full((idx.shape[0], T.size), np.nan)
    status = np.empty((idx.shape[0], T.size), dtype=object)
    for b, rows in enumerate(idx):
        reps[b], status[b] = return_levels_with_status(model._take(rows), T)
    lo, hi, ci_st = np.full(T.size, np.nan), np.full(T.size, np.nan), np.empty(T.size, dtype=object)
    for k in range(T.size):
        lo[k], hi[k], ci_st[k] = percentile_interval(reps[:, k], status[:, k] == OK, ci_level,
                                                     min_valid_count, min_valid_frac)
    return BootstrapResult(T, est, est_status, lo, hi, ci_st, reps, status, idx, float(ci_level))
