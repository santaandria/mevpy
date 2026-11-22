"""Daily-rainfall preprocessing for the annual MEVD (separate from the model).

Adapted from mevpy v1 (E. Zorzetto; ``mev_fit`` threshold excesses,
``remove_missing_years``).

Rules
-----
Data       a daily series with a DatetimeIndex; NaN = missing observation.
Block      calendar year.
Event      a day with depth > ``threshold`` (strict); its magnitude is the
           depth itself.  Use ``WeibullPWM(loc=threshold)`` so the Weibull is
           fitted to the excesses depth - threshold.
Validity   a year is included only if completeness = observed days / calendar
           days of that year >= ``min_completeness`` (default 0.8).  Years
           below it -- including years absent from the record -- are excluded
           from the model, never treated as years without events.
Partial    a year only partly covered by the record (first or last year) is
           judged against its full calendar length, i.e. by the same
           completeness rule as any other year.
Incomplete valid years   n_j counts events on observed days only (it is not
           rescaled), so it is a lower bound of the true count.  The fraction
           of missed events is not bounded by the fraction of missing days:
           if the missing days fall in the wet season, a year with 20 %
           missing days can miss a much larger share of its events.  Raise
           ``min_completeness`` to limit the effect.
Dry valid years          observed years without events (n_j = 0) cannot be
           fitted: ``MEVD.fit`` raises FitError if they are included.
           Excluding them is an explicit choice that conditions the model on
           wet years (return periods then count wet years); see ``core.py``.
Annual maxima (GEV input) maximum of the observed days of each valid year.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


def read_day_by_year_csv(path, day_column="gg"):
    """Read a wide day-by-year table into a daily series (NaN = missing).

    Layout: one row per calendar day labelled ``day_column`` ("1-Jan", ...,
    "29-Feb", ..., "31-Dec"), one column per year ("1725", ...), values are
    daily depths (no unit conversion is applied).  Cells that do not form a
    real date (29-Feb of non-leap years) are dropped; a warning reports any
    non-zero value lost that way, since it usually indicates a data error.
    Years without a column, and days missing inside the range, become NaN
    via a daily ``asfreq``.
    """
    wide = pd.read_csv(path)
    if day_column not in wide.columns:
        raise ValueError(f"{path}: day column {day_column!r} not found")
    long = wide.melt(id_vars=day_column, var_name="year", value_name="prcp")
    date = pd.to_datetime(long[day_column] + "-" + long["year"].astype(str),
                          format="%d-%b-%Y", errors="coerce")
    lost = long.loc[date.isna() & long["prcp"].fillna(0).ne(0)]
    if len(lost):
        cells = ", ".join(f"{d}-{y}={v:g}" for d, y, v in lost[[day_column, "year", "prcp"]].values[:5])
        warnings.warn(f"{len(lost)} non-zero value(s) on invalid dates dropped: {cells}", stacklevel=2)
    series = pd.Series(long["prcp"].to_numpy(float), index=date)[date.notna().to_numpy()]
    series = series.sort_index()
    if series.index.has_duplicates:
        raise ValueError(f"{path}: duplicated dates")
    return series.asfreq("D").rename("prcp")


def _check_daily(daily):
    if not isinstance(daily, pd.Series) or not isinstance(daily.index, pd.DatetimeIndex):
        raise TypeError("daily must be a pandas Series with a DatetimeIndex")
    if daily.index.has_duplicates:
        raise ValueError("daily series has duplicated timestamps")


def yearly_table(daily, threshold, min_completeness=0.8):
    """One row per calendar year from the first to the last year of ``daily``.

    Columns: n_obs (observed days), n_days (calendar days), completeness,
    valid, n_events (days > threshold), annual_max (NaN if no observation).
    """
    _check_daily(daily)
    if not 0 < min_completeness <= 1:
        raise ValueError("min_completeness must be in (0, 1]")
    years = np.arange(daily.index.year.min(), daily.index.year.max() + 1)
    by_year = daily.groupby(daily.index.year)
    n_obs = by_year.count().reindex(years, fill_value=0)
    n_events = by_year.apply(lambda v: int((v > threshold).sum())).reindex(years, fill_value=0)
    annual_max = by_year.max().reindex(years)
    n_days = pd.Series(np.where(pd.DatetimeIndex(pd.to_datetime(years.astype(str))).is_leap_year,
                                366, 365), index=years)
    table = pd.DataFrame({"n_obs": n_obs.astype(int), "n_days": n_days,
                          "completeness": n_obs / n_days, "n_events": n_events.astype(int),
                          "annual_max": annual_max})
    table.insert(3, "valid", table["completeness"] >= min_completeness)
    table.index.name = "year"
    return table


def ordinary_daily_events(daily, threshold, years):
    """Event depths (> threshold) and their year labels, for the given years.

    Pass the valid years only; use the same ``years`` as ``block_index`` when
    fitting, so that a valid dry year is reported by ``MEVD.fit`` (FitError)
    instead of disappearing unnoticed.
    """
    _check_daily(daily)
    v = daily.to_numpy(float)
    y = daily.index.year.to_numpy()
    with np.errstate(invalid="ignore"):
        keep = (v > threshold) & np.isin(y, np.asarray(years))
    return v[keep], y[keep]
