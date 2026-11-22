"""Empirical Metastatistical Extreme Value Distribution (MEVD).

Adapted from mevpy v1 (E. Zorzetto; ``mev_cdf``, ``mev_quant``):
object-oriented rewrite with a generalised ordinary distribution and a new
quantile inversion.

    H(x) = (1 / M) * sum_j F(x; theta_j) ** n_j

M is the number of included blocks, n_j >= 1 the number of ordinary events in
block j and theta_j the ordinary-distribution parameters fitted to that
block's events alone.  The model stores the paired arrays ``n`` and ``params``
(one row per block) together with the ordinary distribution; it never pools
events across blocks or averages parameters.

Blocks
------
A block is whatever ``blocks`` labels say (a year for annual maxima).  The
included blocks are ``block_index`` if given, otherwise the distinct labels
present.  Every included block must contain at least one event.

Zero-event blocks (n_j = 0, e.g. an observed year without any event) are
rejected with an error, in ``fit`` and in direct construction; they are never
dropped silently.  Leaving genuine zero-event blocks out (an explicit choice
of the caller) conditions the model on event-containing blocks: H is then the
distribution of the block maximum given at least one event, and return
periods count event-containing blocks, not all blocks.  With a fraction p0 of
zero-event blocks, the unconditional exceedance probability of a level is
(1 - p0) times the conditional one.

Missing or incomplete blocks are a different matter: their events are
unknown, not absent.  They must be left out of both the events and
``block_index`` (see ``rainfall.py`` for the annual rainfall rules); this does
not condition on event occurrence as long as missingness is unrelated to it.

Insufficient samples
--------------------
A block with n_j < ``distribution.min_events``, or whose fit raises or returns
non-finite values, makes ``fit`` raise :class:`FitError` naming the block.
Such a block is not dropped automatically, because dropping a block changes M
and biases H; decide explicitly (e.g. exclude it from ``block_index``) if that
is acceptable.

Stored state
------------
The constructor copies its inputs: ``blocks``, ``n`` and ``params`` are
read-only arrays and ``distribution`` is a private deep copy, so later changes
to the objects passed in cannot desynchronise the model.  The distribution
object must not be modified in place through ``model.distribution``.

Quantiles and endpoints
-----------------------
Each F_j must be continuous and strictly increasing between its support
endpoints (``distribution.support``).  With a = min_j lower_j and
b = max_j upper_j: ``cdf`` follows the formula for every x; ``ppf(0)`` = a,
``ppf(1)`` = b, NaN for q outside [0, 1] or NaN (as scipy), and otherwise the
generalised inverse inf{x : H(x) >= q}.  Where H is flat (gaps between the
supports of different blocks) this is the left end of the flat interval.
Failed inversion raises :class:`InversionError`.

Scalar inputs return numpy scalars, array inputs arrays of the same shape.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp


class FitError(RuntimeError):
    """The MEVD (one of its block fits) could not be estimated."""


class InversionError(RuntimeError):
    """A quantile of the MEVD could not be computed to the required accuracy."""


def _out(a):
    """0-d arrays -> numpy scalar, others unchanged (scipy convention)."""
    return a[()] if a.ndim == 0 else a


def _zero_event_message(labels):
    return (f"zero-event block(s) {labels[:10]}: every block must contain at least one event. "
            "Zero-event blocks are not supported and are never dropped silently; excluding "
            "genuine zero-event blocks conditions the model on event-containing blocks "
            "(see the documentation on blocks).")


def _readonly(a):
    a.setflags(write=False)
    return a


class MEVD:
    """Fitted MEVD; behaves like a frozen ``scipy.stats`` distribution.

    Build it with :meth:`MEVD.fit`, or directly from stored parameters:
    ``MEVD(distribution, blocks, n, params)`` with ``n`` >= 1 and ``params``
    of shape ``(M, k)``.  Inputs are copied.

    Attributes (read-only): ``distribution``, ``blocks`` (labels, shape (M,)),
    ``n`` (event counts, (M,)), ``params`` ((M, k)), ``n_blocks`` (= M).
    """

    rtol_residual = 1e-6        # max relative error of the probability at a quantile
    max_expand = 60             # bracket expansion steps (guard against rounding)

    def __init__(self, distribution, blocks, n, params):
        self._setup(copy.deepcopy(distribution), blocks, n, params)

    def _setup(self, distribution, blocks, n, params):
        """Validate, copy and cache.  ``distribution`` is stored as given."""
        n = np.array(n)
        params = np.array(params, dtype=float, ndmin=2)
        blocks = np.array(blocks)
        if n.ndim != 1 or n.size == 0:
            raise ValueError("n must be a non-empty 1-D array (one count per block)")
        if not (np.issubdtype(n.dtype, np.integer) or np.all(n == np.round(n))) or np.any(n < 0):
            raise ValueError("n must contain non-negative integers")
        if params.ndim != 2 or params.shape[0] != n.size or blocks.shape != n.shape:
            raise ValueError(f"blocks {blocks.shape}, n {n.shape} and params {params.shape} "
                             "must have one entry/row per block")
        if np.any(n == 0):
            raise ValueError(_zero_event_message(blocks[n == 0].tolist()))
        if not np.all(np.isfinite(params)):
            raise ValueError("params must be finite for every block")
        self._distribution = distribution
        self._blocks = _readonly(blocks)
        self._n = _readonly(n.astype(int))
        self._params = _readonly(params)
        self._n_f = self._n.astype(float)
        self._theta = tuple(self._params.T)
        lo, hi = distribution.support(*self._theta)
        self._lower = np.broadcast_to(np.asarray(lo, dtype=float), self._n.shape).copy()
        self._upper = np.broadcast_to(np.asarray(hi, dtype=float), self._n.shape).copy()
        self._a = float(np.min(self._lower))
        self._b = float(np.max(self._upper))

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, values, blocks, distribution, block_index=None):
        """Fit theta_j separately to the events of each block.

        values      : 1-D array of event magnitudes (finite).
        blocks      : 1-D array of block labels, same length as ``values``.
        distribution: ordinary distribution (see ``mevpy.ordinary``).
        block_index : labels of all included blocks, in the order to store
                      them (default: sorted distinct labels of ``blocks``).
                      Every event's label must be listed, and every listed
                      block must contain at least one event (a listed block
                      without events raises FitError).
        """
        values = np.asarray(values, dtype=float)
        blocks = np.asarray(blocks)
        if values.ndim != 1 or blocks.shape != values.shape:
            raise ValueError("values and blocks must be 1-D arrays of the same length")
        if not np.all(np.isfinite(values)):
            raise ValueError("values contain NaN or inf; remove missing observations before "
                             "fitting (they are not zero-event periods)")
        if block_index is None:
            block_index = np.unique(blocks)
        else:
            block_index = np.asarray(block_index)
            if block_index.ndim != 1 or np.unique(block_index).size != block_index.size:
                raise ValueError("block_index must be 1-D with unique labels")
            unknown = np.setdiff1d(np.unique(blocks), block_index)
            if unknown.size:
                raise ValueError(f"events in blocks not listed in block_index: {unknown[:10].tolist()}")
        if block_index.size == 0:
            raise FitError("no blocks to fit")
        sorter = np.argsort(block_index)
        codes = sorter[np.searchsorted(block_index, blocks, sorter=sorter)]
        n = np.bincount(codes, minlength=block_index.size)
        if np.any(n == 0):
            raise FitError(_zero_event_message(block_index[n == 0].tolist()))
        rows = []
        for j, label in enumerate(block_index):
            if n[j] < distribution.min_events:
                raise FitError(f"block {label!r} has {n[j]} event(s); "
                               f"{distribution!r} needs at least {distribution.min_events}")
            try:
                theta = np.asarray(distribution.fit(values[codes == j]), dtype=float).ravel()
            except Exception as exc:
                raise FitError(f"fit failed in block {label!r} (n={n[j]}): {exc}") from exc
            if not np.all(np.isfinite(theta)):
                raise FitError(f"fit in block {label!r} returned non-finite parameters {theta}")
            rows.append(theta)
        k = {r.size for r in rows}
        if len(k) != 1:
            raise FitError(f"fits returned parameter vectors of different lengths {k}")
        return cls(distribution, block_index, n, np.vstack(rows))

    # ------------------------------------------------------------ utilities
    @property
    def distribution(self):
        return self._distribution

    @property
    def blocks(self):
        return self._blocks

    @property
    def n(self):
        return self._n

    @property
    def params(self):
        return self._params

    @property
    def n_blocks(self):
        return self._n.size

    def _take(self, idx):
        """MEVD of the blocks at positions ``idx`` (repeats = separate blocks).
        Shares this model's private distribution copy."""
        idx = np.asarray(idx, dtype=int)
        new = MEVD.__new__(MEVD)
        new._setup(self._distribution, self._blocks[idx], self._n[idx], self._params[idx])
        return new

    def to_frame(self):
        """Per-block parameters as a pandas DataFrame (block, n, theta...); a copy."""
        import pandas as pd
        names = list(getattr(self.distribution, "param_names", ()))
        if len(names) != self.params.shape[1]:
            names = [f"theta{i}" for i in range(self.params.shape[1])]
        df = pd.DataFrame(self.params, columns=names, copy=True)
        df.insert(0, "n", self.n.copy())
        df.insert(0, "block", self.blocks.copy())
        return df

    def __repr__(self):
        return f"MEVD({self.distribution!r}, n_blocks={self.n_blocks}, mean_n={self.n.mean():.4g})"

    # -------------------------------------------------- distribution methods
    def _log_f(self, x):
        """log F(x; theta_j) for every block, shape x.shape + (M,)."""
        return self.distribution.logcdf(x[..., None], *self._theta)

    def logcdf(self, x):
        x = np.asarray(x, dtype=float)
        with np.errstate(divide="ignore"):
            out = logsumexp(self._n_f * self._log_f(x), axis=-1) - np.log(self.n_blocks)
        return _out(np.where(np.isnan(x), np.nan, out))

    def cdf(self, x):
        return _out(np.exp(self.logcdf(x)))

    def sf(self, x):
        """1 - H(x), computed as mean_j(-expm1(n_j log F_j)): accurate when tiny,
        provided the distribution's logcdf is accurate near 0."""
        x = np.asarray(x, dtype=float)
        out = -np.expm1(self._n_f * self._log_f(x)).sum(axis=-1) / self.n_blocks
        return _out(np.where(np.isnan(x), np.nan, out))

    def logsf(self, x):
        with np.errstate(divide="ignore"):
            return _out(np.log(self.sf(x)))

    def support(self):
        return self._a, self._b

    def ppf(self, q):
        """Quantile inf{x : H(x) >= q} (see module docstring for endpoints)."""
        q = np.asarray(q, dtype=float)
        out = np.array([self._quantile(1.0 - v, v) for v in q.ravel()]).reshape(q.shape)
        return _out(out)

    def isf(self, s):
        """Inverse survival function: ppf(1 - s), computed from s (accurate for tiny s)."""
        s = np.asarray(s, dtype=float)
        out = np.array([self._quantile(v, 1.0 - v) for v in s.ravel()]).reshape(s.shape)
        return _out(out)

    def median(self):
        return self.ppf(0.5)

    def interval(self, confidence):
        c = float(confidence)
        return self.ppf((1.0 - c) / 2.0), self.ppf((1.0 + c) / 2.0)

    def return_level(self, T):
        """Level exceeded on average once every T blocks: ppf(1 - 1/T), T > 1.

        T is in units of the block duration (years only for annual blocks).
        Computed as isf(1/T), which avoids rounding 1 - 1/T for large T.
        """
        T = np.asarray(T, dtype=float)
        if not np.all(np.isfinite(T) & (T > 1)):
            raise ValueError("return periods must be finite and > 1 (in block units)")
        return self.isf(1.0 / T)

    # ------------------------------------------------------------ inversion
    def _block_isf(self, t):
        """Per-block level x_j with 1 - F_j(x_j) ** n_j = t (0 < t < 1)."""
        sf_f = -np.expm1(np.log1p(-t) / self._n_f)
        return self.distribution.isf(sf_f, *self._theta)

    def _block_ppf(self, t):
        """Per-block level x_j with F_j(x_j) ** n_j = t (0 < t < 1)."""
        return self.distribution.ppf(np.exp(np.log(t) / self._n_f), *self._theta)

    def _quantile(self, s, q):
        """inf{x : H(x) >= q} with s = 1 - q.  The caller passes both, the one
        it received exactly; the tail is chosen first and only the exact (the
        smaller) probability is used from then on.  Raises InversionError."""
        if not (0 <= q <= 1 and 0 <= s <= 1):          # also NaN
            return np.nan
        m = self.n_blocks
        # g is decreasing in x, with g(x) == 0 at the quantile.
        # Bracket valid in exact arithmetic, from per-block levels x_j(t):
        #   G_j(x) = F_j(x)^n_j and H = mean_j G_j.
        if q < 0.5:                                   # lower tail: work with q
            if q == 0:
                return self._a
            log_q = np.log(q)

            def g(x):
                return log_q - self.logcdf(x)
            with np.errstate(all="ignore"):
                # lo: every G_j(lo) <= q.  hi: one block alone gives H >= q.
                lo = float(np.min(self._block_ppf(q)))
                hi = (float(np.min(self._block_ppf(m * q))) if m * q < 1
                      else float(np.max(self._block_ppf(q))))
        else:                                         # upper tail: work with s
            if s == 0:
                return self._b
            log_s = np.log(s)

            def g(x):
                return self.logsf(x) - log_s
            with np.errstate(all="ignore"):
                # hi: every 1 - G_j(hi) <= s.  lo: one block alone gives 1 - H >= s.
                hi = float(np.max(self._block_isf(s)))
                lo = (float(np.max(self._block_isf(m * s))) if m * s < 1
                      else float(np.min(self._block_isf(s))))
        p = f"q={q:.6g}" if q < 0.5 else f"s={s:.6g}"
        if not (np.isfinite(lo) and np.isfinite(hi)):
            raise InversionError(f"non-finite starting bracket [{lo}, {hi}] for {p}")
        g_lo, g_hi = g(lo), g(hi)
        a, b = self._a, self._b
        for _ in range(self.max_expand):        # rounding guards only (scale-free steps)
            if g_hi <= 0:
                break
            step = max(abs(hi), hi - lo) or 1.0
            hi = hi + (b - hi) / 2 if np.isfinite(b) else hi + step
            g_hi = g(hi)
        for _ in range(self.max_expand):
            if g_lo >= 0 or lo <= a:
                break
            step = max(abs(lo), hi - lo) or 1.0
            lo = a + (lo - a) / 2 if np.isfinite(a) else lo - step
            g_lo = g(lo)
        if not (np.isfinite(g_lo) and np.isfinite(g_hi)) or g_lo < 0 or g_hi > 0:
            raise InversionError(f"could not bracket the quantile for {p} "
                                 f"(lo={lo:.6g}, g={g_lo:.3g}; hi={hi:.6g}, g={g_hi:.3g})")
        if g_lo == 0:
            x = lo
        elif g_hi == 0:
            x = hi
        else:
            xtol = 1e-14 * max(abs(lo), abs(hi), 1e-300)
            x, res = brentq(g, lo, hi, xtol=xtol, rtol=4 * np.finfo(float).eps,
                            maxiter=200, full_output=True, disp=False)
            if not res.converged:
                raise InversionError(f"root search did not converge for {p}")
        # Generalised inverse: a root outside every block's open support lies
        # on a flat interval of H; its left end is the nearest upper endpoint.
        if not np.any((self._lower < x) & (x < self._upper)):
            left = self._upper[self._upper <= x]
            if left.size:
                x = float(left.max())
        resid = abs(np.expm1(g(x)))             # relative error of the probability
        if not resid <= self.rtol_residual:
            raise InversionError(f"quantile residual {resid:.3g} > {self.rtol_residual} for {p}")
        return float(x)
