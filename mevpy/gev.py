"""GEV distribution of block maxima, used as the benchmark for the MEVD.

The L-moment fit is adapted from mevpy v1 (E. Zorzetto;
``gev_fit_lmom``), with the shape solved exactly; the MLE is new.

Two estimators are provided; ``fit_gev(amax, method=...)`` selects one.

``"lmom"`` (default) -- method of L-moments (Hosking, Wallis & Wood 1985).
    The unbiased sample L-moments l1, l2 and t3 = l3 / l2 are matched to
    those of the GEV.  The shape is the root of tau3(k) = t3 found by a
    bracketed Brent search to machine precision (tau3 decreases from 1 at
    k = -1 to -1 as k -> inf), not Hosking's rational approximation.
    L-moments exist only for xi < 1 (k > -1); the root is searched for
    -1 < k <= 50, and samples whose t3 lies outside tau3 of that range are
    reported as ``gev_lmom_no_solution``.  Unlike the MLE, an L-moment fit may
    place the smallest observations below the fitted lower bound (xi > 0);
    this is a property of the estimator and is accepted.  The estimator needs
    no iterative optimiser and so has no optimiser convergence failures.

``"mle"`` -- maximum likelihood, multi-start with diagnostic checks.
    Nelder-Mead minimisation of scipy's penalised negative log-likelihood,
    started from scipy's default start (giving exactly
    ``scipy.stats.genextreme.fit``), from Gumbel moment estimates and from
    L-moment estimates.  A candidate is discarded if Nelder-Mead did not
    converge, the parameters are invalid, an observation lies outside the
    fitted support, xi <= -1 (c >= 1, where no MLE exists; Smith 1985), or a
    single profile-likelihood check at twice the fitted xi finds a higher
    likelihood (a symptom of the ridge xi -> +inf that tied sample minima
    produce, e.g. in bootstrap resamples).  Of the remaining candidates the
    one with the lowest negative log-likelihood is used; scipy's own result
    is preferred whenever it is within ``nll_tol`` of it.  No remaining
    candidate -> invalid.  These checks are heuristics: they do not prove that
    the result is a local, let alone the global, maximum of the likelihood --
    only that it is the best of the three local searches that passed them.

Parameters follow scipy: (c, loc, scale) with c = -xi (c equals Hosking's k).
Failures are reported in ``GevFit.status``, never raised.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy import special as sc
from scipy.optimize import brentq
from scipy.special import gamma
from scipy.stats import genextreme

OK = "ok"
_NAN3 = (np.nan, np.nan, np.nan)
_XMIN = np.finfo(float).tiny
_LOGXMAX = np.log(np.finfo(float).max)


@dataclass
class GevFit:
    params: tuple                 # (c, loc, scale), c = -xi
    status: str                   # "ok" or the reason the fit is invalid
    nll: float = np.nan           # MLE only
    start: str = ""               # MLE: start that produced the accepted fit; "lmom" otherwise
    default_status: str = ""      # MLE: outcome of scipy's default start


def fit_gev(amax, method="lmom"):
    """Fit a GEV to block maxima with ``method`` "lmom" (default) or "mle"."""
    if method == "lmom":
        return fit_gev_lmom(amax)
    if method == "mle":
        return fit_gev_mle(amax)
    raise ValueError(f"unknown GEV method {method!r}; use 'lmom' or 'mle'")


def _check_sample(amax):
    data = np.asarray(amax, dtype=float).ravel()
    return data if data.size >= 3 and np.isfinite(data).all() else None


def _sample_lmoments(data):
    """Unbiased sample L-moments l1, l2, l3 (via probability-weighted moments)."""
    x = np.sort(data)
    n = x.size
    i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n
    return b0, 2 * b1 - b0, 6 * b2 - 6 * b1 + b0


# ---------------------------------------------------------------- L-moments
def _tau3(k):
    """L-skewness of a GEV with shape k (= scipy c), valid for k > -1."""
    if k == 0.0:
        return 2.0 * np.log(3.0) / np.log(2.0) - 3.0                 # Gumbel limit
    return 2.0 * np.expm1(-k * np.log(3.0)) / np.expm1(-k * np.log(2.0)) - 3.0


def fit_gev_lmom(amax):
    """GEV by the method of L-moments (see module docstring)."""
    data = _check_sample(amax)
    if data is None:
        return GevFit(_NAN3, "gev_insufficient_data", start="lmom")
    l1, l2, l3 = _sample_lmoments(data)
    if not l2 > 0:                                                    # constant sample
        return GevFit(_NAN3, "gev_invalid_params", start="lmom")
    t3 = l3 / l2
    k_lo, k_hi = -1.0 + 1e-9, 50.0
    if not _tau3(k_hi) < t3 < _tau3(k_lo):
        return GevFit(_NAN3, "gev_lmom_no_solution", start="lmom")
    k = brentq(lambda k: _tau3(k) - t3, k_lo, k_hi, xtol=1e-14, rtol=4 * np.finfo(float).eps)
    # scale = l2 k / ((1 - 2^-k) Gamma(1+k)); loc = l1 - scale (1 - Gamma(1+k)) / k,
    # written with expm1/gammaln so the k -> 0 (Gumbel) limit is accurate.
    if k == 0.0:
        k_ratio, g_ratio = 1.0 / np.log(2.0), np.euler_gamma
    else:
        k_ratio = k / -np.expm1(-k * np.log(2.0))
        g_ratio = -np.expm1(sc.gammaln(1.0 + k)) / k
    scale = l2 * k_ratio / gamma(1.0 + k)
    params = (float(k), float(l1 - scale * g_ratio), float(scale))
    ok = np.all(np.isfinite(params)) and scale > 0
    return GevFit(params, OK if ok else "gev_invalid_params", start="lmom")


# ----------------------------------------------------------------------- MLE
def _gev_nll(theta, data):
    """scipy's ``genextreme._penalized_nnlf`` for 1-D data, without the generic
    broadcasting overhead (same operations, same values)."""
    c, loc, scale = theta
    if not np.isfinite(c) or scale <= 0:
        return np.inf
    x = (data - loc) / scale
    n_log_scale = len(x) * np.log(scale)
    b = 1.0 / max(c, _XMIN) if c > 0 else np.inf
    a = 1.0 / min(c, -_XMIN) if c < 0 else -np.inf
    with np.errstate(invalid="ignore"):
        inside = (a <= x) & (x <= b)
    n_bad = np.count_nonzero(~inside)
    if n_bad > 0:
        x = x[inside]
    with np.errstate(all="ignore"):
        if c != 0:
            cx = c * x
            logpex2 = sc.log1p(-c * x) / c
        else:
            cx = np.zeros_like(x)
            logpex2 = -x
        logex2 = sc.log1p(-cx)
        pex2 = np.exp(logpex2)
        logpdf = np.where(~((cx == 1) | (cx == -np.inf)), -pex2 + logpex2 - logex2, -np.inf)
    logpdf[(c == 1) & (x == 1)] = 0.0
    fin = np.isfinite(logpdf)
    n_bad += fin.size - np.count_nonzero(fin)
    return -np.sum(logpdf[fin]) + n_bad * _LOGXMAX * 100 + n_log_scale


def _gumbel_start(data):
    scale = np.std(data) * np.sqrt(6.0) / np.pi
    return (0.0, np.mean(data) - np.euler_gamma * scale, scale)


def _lmoment_start(data):
    """L-moment estimate with Hosking's (1985) approximation for the shape."""
    b0, l2, l3 = _sample_lmoments(data)
    z = 2.0 / (3.0 + l3 / l2) - np.log(2) / np.log(3)
    k = 7.8590 * z + 2.9554 * z * z
    scale = l2 * k / ((1 - 2.0 ** -k) * gamma(1 + k))
    return (k, b0 - scale * (1 - gamma(1 + k)) / k, scale)


def _fit_from(data, start, max_restarts):
    """One Nelder-Mead run; start None = scipy's own start (genextreme.fit)."""
    try:
        x0 = genextreme._fitstart(data) if start is None else start
        theta, fopt, _, _, flag = optimize.fmin(_gev_nll, x0, args=(data,), disp=0, full_output=True)
        restarts = 0
        while flag != 0 and restarts < max_restarts:     # continue from where it stopped
            restarts += 1
            theta, fopt, _, _, flag = optimize.fmin(_gev_nll, theta, args=(data,), disp=0,
                                                    full_output=True)
    except Exception:
        return GevFit(_NAN3, "gev_fit_error")
    theta = tuple(float(t) for t in theta)
    c, loc, scale = theta
    fit = GevFit(theta, OK, float(fopt), start="default")
    if not (np.all(np.isfinite(theta)) and scale > 0):
        fit.status = "gev_invalid_params"
    elif c >= 1:
        fit.status = "gev_unbounded_likelihood"          # xi <= -1: no MLE
    elif flag != 0:
        fit.status = "gev_not_converged"
    elif not np.isfinite(genextreme.logpdf(data, c, loc, scale)).all():
        fit.status = "gev_data_outside_support"
    return fit


def _likelihood_grows_with_xi(data, fit, nll_tol):
    """True if the profile likelihood is higher at twice the fitted xi.

    A single-point check: True shows the candidate is not a maximum along xi;
    False does not prove that it is one.
    """
    c, loc, scale = fit.params
    with np.errstate(all="ignore"):
        x = optimize.fmin(lambda p: _gev_nll(np.array([2 * c, p[0], p[1]]), data),
                          [loc, 2 * scale], disp=0)
        return _gev_nll(np.array([2 * c, x[0], x[1]]), data) < fit.nll - nll_tol


def fit_gev_mle(amax, max_restarts=2, nll_tol=1e-3):
    """Multi-start GEV maximum likelihood with diagnostic checks (see module docstring)."""
    data = _check_sample(amax)
    if data is None:
        return GevFit(_NAN3, "gev_insufficient_data")
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        default = _fit_from(data, None, max_restarts)
        candidates = [default]
        for name, start_fun in (("gumbel", _gumbel_start), ("lmoments", _lmoment_start)):
            try:
                x0 = start_fun(data)
            except Exception:
                continue
            if np.all(np.isfinite(x0)) and x0[2] > 0:
                cand = _fit_from(data, x0, max_restarts)
                cand.start = name
                candidates.append(cand)
        best = None
        for f in sorted((f for f in candidates if f.status == OK), key=lambda f: f.nll):
            if f.params[0] < 0 and _likelihood_grows_with_xi(data, f, nll_tol):
                f.status = "gev_unbounded_likelihood"
                continue
            best = f
            break
        if (best is not None and best is not default and default.status == OK
                and default.nll <= best.nll + nll_tol):
            if default.params[0] < 0 and _likelihood_grows_with_xi(data, default, nll_tol):
                default.status = "gev_unbounded_likelihood"
            else:
                best = default
    if best is None:
        default.default_status = default.status
        return default
    best.default_status = default.status if default.status != OK else (
        OK if best is default else "gev_default_not_mle")
    return best


# ------------------------------------------------------------- return levels
def gev_return_levels(fit, return_periods, rtol_residual=1e-6):
    """Return levels isf(1/T) and per-level status for a GevFit.

    A level is withheld (NaN, status says why) if the fit is invalid, the
    quantile is not finite, or sf(level) * T differs from 1 by > rtol_residual.
    """
    s = 1.0 / np.atleast_1d(np.asarray(return_periods, dtype=float))
    rl = np.full(s.shape, np.nan)
    status = np.full(s.shape, fit.status, dtype=object)
    if fit.status != OK:
        return rl, status
    with np.errstate(all="ignore"):
        q = genextreme.isf(s, *fit.params)
        resid = np.abs(genextreme.sf(q, *fit.params) / s - 1.0)
    for k in range(s.size):
        if not np.isfinite(q[k]):
            status[k] = "gev_quantile_nonfinite"
        elif not resid[k] <= rtol_residual:
            status[k] = "gev_quantile_residual"
        else:
            status[k], rl[k] = OK, q[k]
    return rl, status
