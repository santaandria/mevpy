# About mevpy

`mevpy` is a small Python package for the Metastatistical Extreme Value Distribution (MEVD) of block maxima (Marani & Ignaccolo 2015; Zorzetto, Botter & Marani 2016):

    H(x) = (1 / M) * sum_j F(x; theta_j) ** n_j

M is the number of included blocks (e.g. years), n_j the number of ordinary events in block j, and theta_j the parameters of the ordinary-event distribution F fitted to block j alone. A fitted model stores the paired arrays `n` and `params` together with the ordinary `distribution`. Events are never pooled across blocks and parameters are never averaged: the inter-block variability of (n_j, theta_j) is what the MEVD represents.

## Origin and credits

This is version 2 of [mevpy](https://github.com/EnricoZorzetto/mevpy), originally written by Enrico Zorzetto (v1.0.6) and distributed under the MIT License (see `LICENSE`).

The methods and the rainfall application come from the original mevpy:
- the MEV formulation, its CDF and quantile;
- the yearly Weibull fit by probability-weighted moments;
- the threshold-excess definition of ordinary events and the removal of incomplete years;
- the GEV fit by L-moments;
- the sliding-window comparison of MEV and GEV.

Version 2 reorganises and extends that code. The API is new and not backward compatible with v1:
- **Object-oriented API.** The `MEVD` class behaves like a frozen `scipy.stats` distribution and stores the paired per-block (n_j, theta_j).
- **Generalisation.** Any ordinary distribution can be used through a small interface, including `scipy.stats` distributions. Blocks need not be years.
- **Quantile inversion.** A Brent root search in log-probability replaces the solver started from an initial guess. It starts from a bracket built from per-block quantiles and ends with an independent residual check. Very large return periods and very small lower-tail probabilities are resolved to the tolerance of that check, and an error is raised when the check fails.
- **Estimators.** The GEV L-moment shape is solved exactly instead of with Hosking's rational approximation. A multi-start maximum-likelihood GEV with diagnostic checks is added.
- **Explicit data rules.** Missing or incomplete years are excluded, never counted as years without events. Insufficient samples raise errors that name the block.
- **Bootstrap.** Whole years are resampled, keeping each year's (n_j, theta_j) pair, with reproducible random streams and failed replicates reported.
- **Tests and documentation.**

## Features

The package provides:
- a fitted `MEVD` object that behaves like a frozen `scipy.stats` distribution (`cdf`, `sf`, `ppf`, `isf`, return levels);
- a small interface for user-defined ordinary distributions, and an adapter for `scipy.stats` distributions;
- the Weibull ordinary distribution fitted by probability-weighted moments (PWM), as used for daily rainfall;
- a nonparametric block bootstrap for return-level confidence intervals;
- daily-rainfall preprocessing, and a GEV benchmark (L-moments or maximum likelihood) for comparing MEV and GEV.

| Module | Content |
|---|---|
| `mevpy/core.py` | `MEVD` (fitting, cdf/sf/ppf/isf, return levels), `FitError`, `InversionError` |
| `mevpy/ordinary.py` | ordinary-distribution interface `OrdinaryDistribution`, `ScipyOrdinary` adapter |
| `mevpy/weibull.py` | `WeibullPWM`: two-parameter Weibull above a threshold, fitted by PWM |
| `mevpy/bootstrap.py` | `bootstrap`, `BootstrapResult`, `percentile_interval` |
| `mevpy/rainfall.py` | daily-rainfall preprocessing: CSV reader, yearly completeness, event extraction |
| `mevpy/gev.py` | GEV benchmark: `fit_gev` (L-moments by default, or multi-start MLE), `gev_return_levels` |
| `examples/rainfall_example.py` | complete rainfall application, including the sliding-window MEV vs GEV comparison |
| `tests/test_mevpy.py` | unit tests |

## Installation

Requires Python ≥ 3.9 with numpy and scipy. The rainfall tools and the example also need pandas and matplotlib.

With conda or mamba, `environment.yml` creates an environment named `mevpy`. It contains all dependencies (conda-forge) and installs `mevpy` from the repository in editable mode:

```bash
mamba env create -f environment.yml      # or: conda env create -f environment.yml
mamba activate mevpy                      # or: conda activate mevpy
```

Run this from the repository root. To update an existing environment after the dependencies change, use `mamba env update -f environment.yml --prune`.

With pip:

```bash
pip install .                 # core
pip install ".[rainfall]"     # with pandas and matplotlib
```

## Minimal usage

```python
from mevpy import MEVD, WeibullPWM, bootstrap

# values: event magnitudes; blocks: the block label of each event (e.g. its year)
# block_index: all included blocks; every block must contain at least one event
model = MEVD.fit(values, blocks, distribution=WeibullPWM(loc=1.0), block_index=valid_years)

model.cdf(100.0)                    # H(100)
model.ppf([0.5, 0.9, 0.99])         # quantiles
model.return_level([10, 100])       # = ppf(1 - 1/T), T in block units
model.sf(150.0), model.isf(1e-3), model.support(), model.median()
model.to_frame()                    # block, n, C, W (one row per block)

res = bootstrap(model, [10, 100], n_boot=1000, ci_level=0.90, random_state=0)
res.summary()                       # estimate, lower, upper, ci_status, n_valid per T
res.failures()                      # every failed (replicate, T) with its error message
res.indices                         # drawn block positions (n_boot, M)
```

A model can also be built directly from stored parameters: `MEVD(distribution, blocks, n, params)`, with all n_j ≥ 1 and `params` of shape (M, k). Inputs are copied. `blocks`, `n` and `params` are exposed as read-only arrays, and the model keeps a private copy of the distribution object.

Return periods are expressed in units of the block duration: they are years only when the blocks are years.

## Input assumptions and conventions

- **Events.** Events are assumed to be already identified and independent within each block, conditional on that block's parameters. The package does no event detection, declustering or independence testing.
- **Blocks.** A block is whatever the `blocks` labels say; for rainfall it is a calendar year. The included blocks are `block_index` if given, otherwise the distinct labels present. Every event label must appear in `block_index`, and every included block must contain at least one event.
- **Zero-event blocks.** A fully observed block without events (n_j = 0, e.g. a dry year) is rejected with an error, both by `fit` and by direct construction; it is never dropped silently.
  - Leaving genuine zero-event blocks out is an explicit choice that conditions the model on event-containing blocks. H is then the distribution of the block maximum given at least one event, and return periods count event-containing blocks, not all blocks.
  - With a fraction p0 of zero-event blocks, the unconditional exceedance probability of a level is (1 − p0) times the conditional one.
- **Missing observations.** `values` must be finite; NaN raises an error. Missing or incomplete blocks differ from zero-event blocks: their events are unknown, not absent. Leave them out of both the events and `block_index`. This does not condition the model on event occurrence, as long as missingness is unrelated to it. See [Rainfall preprocessing](#rainfall-preprocessing) for the annual rules.
- **Insufficient samples.** If a block has n_j < `distribution.min_events` (2 for PWM), `fit` raises `FitError` naming the block. It does the same when a block's fit fails or returns non-finite parameters. Blocks are never dropped silently, because dropping one changes M.
- **Endpoints and quantiles.** Let a and b be the smallest lower and the largest upper support endpoints over all blocks. They come from `distribution.support`, i.e. from the distribution's support, not from the observed sample range.
  - `cdf` follows the formula for every x.
  - `ppf(0)` returns a, `ppf(1)` returns b, and q outside [0, 1] gives NaN.
  - Otherwise `ppf(q)` returns the generalised inverse inf{x : H(x) ≥ q}. Where H is flat, i.e. in a gap between the supports of different blocks, this is the left end of the gap.
- **Shapes.** Scalar inputs give numpy scalars; arrays keep their shape.
- **Quantile inversion.** Brent's method on log(1 − H) − log(1 − q), or on log q − log H when q < 0.5, so the smaller probability is resolved accurately.
  - The tail is chosen first, and from then on only the smaller of q and 1 − q is used, without forming 1 − q.
  - The starting bracket needs no initial guess. It is valid in exact arithmetic, and a bounded expansion guards against rounding. It comes from the per-block quantiles of G_j = F_j^{n_j}.
    - Upper tail, with s = 1 − q and x_j(t) solving 1 − G_j = t: the upper bound is max_j x_j(s). The lower bound is max_j x_j(M s) if M s < 1, otherwise min_j x_j(s).
    - Lower tail, with x_j(t) solving G_j = t: the lower bound is min_j x_j(q). The upper bound is min_j x_j(M q) if M q < 1, otherwise max_j x_j(q).
  - 1 − F^n is computed as `-expm1(n * logF)`. This keeps tiny exceedance probabilities (large T) accurate, provided the distribution's `logcdf` is accurate.
  - Every root is checked independently. An `InversionError` is raised if the relative error of the probability exceeds 1e-6, the bracket cannot be established, or the root search does not converge.

## Custom ordinary distributions

Subclass `OrdinaryDistribution`, or provide an object with the same members:

| Member | Contract |
|---|---|
| `fit(sample) -> tuple` | Estimates theta from **one block's** events (1-D float array, length ≥ `min_events`). Raise on failure. |
| `cdf(x, *theta)` | F(x; theta), continuous and strictly increasing between the support endpoints. Must broadcast like a numpy ufunc: it is called with `x` of shape `(..., 1)` and each parameter of shape `(m,)`, one entry per block. |
| `ppf(p, *theta)` | Inverse CDF, same broadcasting. Used for the inversion bracket. |
| `support(*theta) -> (lo, hi)` | Support endpoints of the distribution, not of the sample (±inf allowed). They define `ppf(0)`, `ppf(1)` and the gaps between block supports. |
| `logcdf`, `isf` (optional) | Defaults are `log(cdf)` and `ppf(1 - s)`. Override them for accurate upper tails. |
| `param_names`, `min_events`, `name` | Column names for `to_frame()`, the minimum block sample size, and a label. |

```python
import numpy as np
from mevpy import MEVD, OrdinaryDistribution, bootstrap

class Exponential(OrdinaryDistribution):
    name, param_names, min_events = "expon", ("scale",), 1
    def fit(self, sample):            return (float(np.mean(sample)),)      # MLE
    def cdf(self, x, scale):          return -np.expm1(-np.maximum(x, 0) / scale)
    def ppf(self, p, scale):          return -scale * np.log1p(-p)
    def support(self, scale):         return np.zeros_like(scale), np.full_like(scale, np.inf)

model = MEVD.fit(values, blocks, Exponential(), block_index=all_blocks)
res = bootstrap(model, [10, 100], n_boot=500, random_state=1)    # same bootstrap workflow
```

A `scipy.stats` distribution can be wrapped directly. Theta is then scipy's `(*shapes, loc, scale)`, and the fit is `dist.fit(sample, **kwargs)`:

```python
from scipy import stats
from mevpy import ScipyOrdinary
model = MEVD.fit(values, blocks, ScipyOrdinary(stats.gamma, floc=0), block_index=all_blocks)
```

### Weibull ordinary distribution (`WeibullPWM`)

`WeibullPWM(loc=threshold)` is the ordinary-event model used for daily rainfall: F(x) = 1 − exp(−((x − loc)/C)^W) for x ≥ loc. C and W are estimated for each block by probability-weighted moments (Greenwood et al. 1979) applied to the excesses x − loc:

    M0 = mean(x),  M1 = sum_i x_(i) (n − i) / (n (n − 1)),  W = ln 2 / ln(M0 / (2 M1)),  C = M0 / Γ(1 + 1/W)

Events at or below `loc` raise an error rather than being dropped, so that n_j and theta_j always describe the same events.

## Bootstrap

`bootstrap()` implements a nonparametric block bootstrap:

- **Resampling unit.** The M included blocks are drawn with replacement. Each draw carries its block's (n_j, theta_j) pair. Repeated draws are separate block instances. Resampling whole blocks assumes blocks are independent and identically distributed, and preserves within-block dependence, including that between n_j and the event magnitudes.
- **Fit.** The replicate MEVD is built from the drawn pairs. This equals refitting the resampled blocks for any deterministic per-block estimator, so custom distributions need nothing extra.
- **Interval.** Percentile interval of the valid replicates, per return period. `ci_level=0.90` gives the 5th and 95th percentiles.
- **Withheld intervals.** An interval is reported only if at least `min_valid_count` (10) replicates and a fraction `min_valid_frac` (0.9) of all replicates are valid. Otherwise it is NaN and `ci_status` gives the reason.
- **Failures.** A failed replicate return level is NaN and its `status` holds the error message. It is excluded only from that return period's interval, and it is counted in `n_valid` and listed by `failures()`.
- **Random state.** `random_state` accepts anything `numpy.random.default_rng` does (None, int, `SeedSequence`, `Generator`). The draws are `rng.integers(0, M, size=(n_boot, M))`. `res.indices` lets other statistics, such as a GEV fit, be computed on the same draws, giving paired intervals.

## GEV benchmark

`mevpy.gev.fit_gev(amax, method=...)` fits a GEV to block maxima for comparison with the MEVD. Parameters follow scipy, `(c, loc, scale)` with c = −ξ. Failures are reported in `GevFit.status`, never raised.

- **`"lmom"` (default): method of L-moments** (Hosking et al. 1985; Hosking 1990). The sample L-moments l1, l2 and t3 are matched: the shape is the root of τ3(k) = t3, found to machine precision with a bracketed solver rather than the usual rational approximation. The root is searched for −1 < k ≤ 50; a sample whose t3 lies outside that range is reported as `gev_lmom_no_solution`. As is inherent to the estimator, the smallest observations may fall below the fitted lower bound when ξ > 0. The method is fast and needs no iterative optimiser, so it has no optimiser convergence failures.
- **`"mle"`: multi-start maximum likelihood with diagnostic checks.** Nelder–Mead is started from scipy's default start and from Gumbel and L-moment starts.
  - A candidate is discarded if it did not converge, if ξ ≤ −1, if data lie outside its support, or if a single profile-likelihood check at twice its ξ finds a higher likelihood. The last check targets the ridge with ξ → +∞ caused by tied minima in bootstrap resamples.
  - The remaining candidate with the highest likelihood is used.
  - These checks are heuristics. They do not prove that the result is the global, or even a local, maximum of the likelihood.

`gev_return_levels(fit, T)` returns the return levels and a status for each one.

## Rainfall preprocessing

`mevpy.rainfall` turns a daily series (pandas `Series` with a `DatetimeIndex`, NaN = missing) into MEVD inputs:

- **Blocks** are calendar years. A year is valid if observed days / calendar days ≥ `min_completeness` (default 0.8). Years below this, including years absent from the record and partly covered first or last years, are excluded; they are never treated as years without events.
- **Events** are days with depth > `threshold`. Use `WeibullPWM(loc=threshold)`, so that the Weibull describes the excesses over the threshold. In a valid but incomplete year, n_j counts events on the observed days only, so it can under-count the year's events. The share of missed events is not bounded by the share of missing days: missing days concentrated in the wet season can hide a much larger share of the events. Raise `min_completeness` to limit the effect.
- **Valid dry years** (observed, n_j = 0) make `MEVD.fit` raise `FitError` (see Zero-event blocks above). Excluding them is an explicit choice that conditions the model on wet years.
- **Annual maxima** for the GEV are the maximum observed depth of each valid year.

```python
from mevpy import MEVD, WeibullPWM, rainfall
daily = rainfall.read_day_by_year_csv("data/pd.csv")          # wide day-by-year table
table = rainfall.yearly_table(daily, threshold=1.0)      # completeness, validity, n_events, annual_max
years = table.index[table["valid"]].to_numpy()
values, blocks = rainfall.ordinary_daily_events(daily, 1.0, years)
model = MEVD.fit(values, blocks, WeibullPWM(loc=1.0), block_index=years)
```

`read_day_by_year_csv` reads a table with one row per calendar day (column `gg`: `1-Jan` … `31-Dec`, including `29-Feb`) and one column per year. Values must be daily depths; no unit conversion is applied. Cells that are not real dates (29 February of non-leap years) are dropped with a warning if they hold a non-zero value.

## Rainfall example

```bash
python examples/rainfall_example.py --data data/pd.csv                   # 500 replicates, all cores
python examples/rainfall_example.py --data data/pd.csv --tr 100 200 --nboot 1000 --n-jobs 16 --outdir examples/output/
python examples/rainfall_example.py --data data/pd.csv --gev-method mle          # GEV by maximum likelihood
python examples/rainfall_example.py --data data/pd.csv --skip-rolling            # single-window demo only
```

The example uses a daily rainfall record from 1725 to 2019 in mm. Fourteen years are missing (1765–67, 1815–22, 1838–40), leaving 281 valid years, none of them dry.

Settings: events are days > 1 mm, blocks are years with ≥ 80 % observed days, windows are 30 years sliding by one year, and a window is evaluated when it has ≥ 20 valid years.

The script:
1. reads and prepares the data;
2. fits the MEVD to the most recent 30-year window and prints the per-year parameters, CDF values and quantiles;
3. computes MEV return levels with bootstrap intervals, next to the GEV;
4. runs the sliding-window comparison of MEV and GEV return levels. Each window's valid years are resampled with replacement, and the same draws are used for both models, so the MEV − GEV difference has a paired interval. The draws of each window come from `SeedSequence(seed, spawn_key=(end_year,))`, so results do not depend on the number of processes.

Outputs (default directory `./mevpy_output/`):

| File | Content |
|---|---|
| `demo_parameters.csv`, `demo_return_levels.csv`, `demo_return_levels.pdf` | single-window fit, return levels with intervals, return-level plot |
| `summary.csv` | one row per (window end year, return period, model ∈ GEV, MEV, MEV−GEV): estimate, interval, statuses, valid and excluded replicates |
| `rolling_Tr<T>.csv` | one row per window for return period T (`gev_*`, `mev_*`, `diff_*` columns) |
| `exclusions.csv` | failed bootstrap estimates per window, return period, model and reason |
| `year_table.csv` | completeness, validity, event count and annual maximum per year |
| `rolling_Tr<T>.pdf` | return levels per window with bootstrap bands |

With L-moments, the full run (261 windows × 500 replicates) takes about 20 s on 14 cores. Maximum likelihood takes several minutes.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover:
- CDF bounds and monotonicity, and agreement between `ppf` and `cdf`, including tiny exceedance probabilities
- the single-block identity H = F^n, and the PWM estimator
- custom and scipy-based ordinary distributions, including in the bootstrap
- bootstrap reproducibility, the resampling unit, the percentile interval, and failure reporting
- the GEV estimators: the L-moment fit reproduces the sample L-moments
- invalid inputs, and the rule that missing years are excluded rather than counted as dry

## References

- Marani, M., & Ignaccolo, M. (2015). A metastatistical approach to rainfall extremes. *Advances in Water Resources*, 79, 121–126. https://doi.org/10.1016/j.advwatres.2015.03.001
- Zorzetto, E., Botter, G., & Marani, M. (2016). On the emergence of rainfall extremes from ordinary events. *Geophysical Research Letters*, 43, 8076–8082. https://doi.org/10.1002/2016GL069445

Estimators:
- Greenwood, J. A., Landwehr, J. M., Matalas, N. C., & Wallis, J. R. (1979). Probability weighted moments: definition and relation to parameters of several distributions expressable in inverse form. *Water Resources Research*, 15(5), 1049–1054.
- Hosking, J. R. M., Wallis, J. R., & Wood, E. F. (1985). Estimation of the generalized extreme-value distribution by the method of probability-weighted moments. *Technometrics*, 27(3), 251–261.
- Hosking, J. R. M. (1990). L-moments: analysis and estimation of distributions using linear combinations of order statistics. *Journal of the Royal Statistical Society B*, 52(1), 105–124.
