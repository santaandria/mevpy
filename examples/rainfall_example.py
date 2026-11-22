"""Daily rainfall example: MEVD fit, CDF and quantiles, bootstrap return levels,
and a sliding-window comparison of MEV and GEV return levels.

The sliding-window comparison follows ``slideover`` of mevpy v1 (E. Zorzetto).

Data
  A wide day-by-year CSV: rows = calendar days (column "gg": "1-Jan" ...
  "31-Dec", including "29-Feb"), one column per year, daily depth in mm.
  Missing years may simply be absent; missing days may be empty cells.

Method
  events     days with depth > --thresh (default 1 mm); the Weibull is fitted
             by probability-weighted moments to depth - thresh, year by year.
  blocks     calendar years with >= --min-completeness (0.8) observed days.
  GEV        fitted to the annual maxima of the same years, by L-moments by
             default (--gev-method mle for multi-start maximum likelihood).
  windows    --ws (30) calendar years sliding by one year, evaluated when they
             contain >= --min-years (20) valid years.
  bootstrap  the valid years of a window are resampled with replacement; the
             same draws are used for MEV and GEV, so their difference has a
             paired interval.  Percentile intervals (--ci-level 0.90).  The
             draws of each window come from SeedSequence(seed, spawn_key=
             (end_year,)), so results do not depend on --n-jobs.

Run
  python examples/rainfall_example.py --data data/pd.csv
  python examples/rainfall_example.py --data data/pd.csv --nboot 1000 --tr 100 200 --n-jobs 16 --outdir examples/output/

Outputs in --outdir (default ./mevpy_output/)
  demo_parameters.csv     per-year (n, C, W) of the most recent window
  demo_return_levels.csv  MEV return levels with bootstrap intervals, and GEV
  demo_return_levels.pdf
  summary.csv             one row per (end_year, Tr, model in GEV / MEV / MEV-GEV):
                          estimate, point_status, ci_l, ci_u, ci_status, n_boot,
                          n_valid, n_excluded, frac_excluded, exclusion_reasons
  rolling_Tr<T>.csv       the same for one return period, one row per window
                          (columns gev_*, mev_*, diff_*)
  exclusions.csv          failed bootstrap estimates per window, Tr, model, reason
  year_table.csv          completeness, validity, event count, annual maximum
  rolling_Tr<T>.pdf       return levels per window with bootstrap bands
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run without installing
from mevpy import MEVD, FitError, WeibullPWM, bootstrap, percentile_interval  # noqa: E402
from mevpy import rainfall  # noqa: E402
from mevpy.bootstrap import OK, bootstrap_indices  # noqa: E402
from mevpy.gev import fit_gev, gev_return_levels  # noqa: E402

COLORS = {"GEV": "#0C5DA5", "MEV": "#FF2C00"}
GEV_LABEL = {"lmom": "GEV (L-moments)", "mle": "GEV (MLE)"}


# --------------------------------------------------------------------------
# One window: MEV and GEV point estimates and paired bootstrap replicates
# --------------------------------------------------------------------------
def analyse_window(task):
    end, years, Tr = task["end_year"], task["years"], task["Tr"]
    nboot, ci = task["nboot"], task["ci"]
    seed = np.random.SeedSequence(task["seed"], spawn_key=(int(end),))
    # MEV
    try:
        model = MEVD.fit(task["values"], task["blocks"], WeibullPWM(task["thresh"]), block_index=years)
        res = bootstrap(model, Tr, n_boot=nboot, ci_level=ci["ci_level"], random_state=seed,
                        min_valid_count=ci["min_valid_count"], min_valid_frac=ci["min_valid_frac"])
        idx = res.indices
        mev = np.vstack([res.estimate, res.replicates])
        mev_st = np.vstack([res.point_status, res.status])
    except FitError as exc:                          # e.g. a valid year with a single event
        idx = bootstrap_indices(years.size, nboot, seed)
        mev = np.full((nboot + 1, Tr.size), np.nan)
        mev_st = np.full((nboot + 1, Tr.size), f"mev_fit_error: {exc}", dtype=object)
    # GEV on the annual maxima of the same years / same draws
    amax = task["amax"]
    gev = np.full_like(mev, np.nan)
    gev_st = np.empty(mev.shape, dtype=object)
    for b, rows in enumerate([np.arange(years.size), *idx]):
        gev[b], gev_st[b] = gev_return_levels(fit_gev(amax[rows], task["gev_method"]), Tr)
    return end, gev, gev_st, mev, mev_st


def _reason(status):
    return str(status).split(":")[0]


def summarise(end, Tr, gev, gev_st, mev, mev_st, ci):
    both = (gev_st == OK) & (mev_st == OK)
    diff_st = np.where(both, OK, np.where(gev_st != OK, gev_st, mev_st))
    rows, excl = [], []
    for model, vals, st in (("GEV", gev, gev_st), ("MEV", mev, mev_st), ("MEV-GEV", mev - gev, diff_st)):
        for k, T in enumerate(Tr):
            boot_st = st[1:, k]
            valid = boot_st == OK
            lo, hi, ci_st = percentile_interval(vals[1:, k], valid, **ci)
            reasons = pd.Series([_reason(s) for s in boot_st[~valid]], dtype=object).value_counts()
            nb = boot_st.size
            for reason, cnt in reasons.items():
                excl.append(dict(end_year=end, Tr=T, model=model, reason=reason, n=int(cnt), frac=cnt / nb))
            rows.append(dict(end_year=end, Tr=T, model=model, estimate=vals[0, k],
                             point_status=_reason(st[0, k]), ci_l=lo, ci_u=hi, ci_status=ci_st,
                             n_boot=nb, n_valid=int(valid.sum()), n_excluded=int(nb - valid.sum()),
                             frac_excluded=(nb - valid.sum()) / nb if nb else np.nan,
                             exclusion_reasons=";".join(f"{a}:{b}" for a, b in reasons.items())))
    return rows, excl


def rolling_comparison(daily, table, ws, min_years, thresh, Tr, nboot, seed, ci, n_jobs,
                       gev_method="lmom"):
    valid = table.index[table["valid"]].to_numpy()
    values, blocks = rainfall.ordinary_daily_events(daily, thresh, valid)
    tasks, rows = [], []
    for end in range(table.index.min() + ws - 1, table.index.max() + 1):
        years = valid[(valid > end - ws) & (valid <= end)]
        if years.size < min_years:
            for T in Tr:
                for model in ("GEV", "MEV", "MEV-GEV"):
                    rows.append(dict(end_year=end, Tr=T, model=model, estimate=np.nan,
                                     point_status="insufficient_valid_years",
                                     ci_status="insufficient_valid_years", n_boot=0, n_valid=0,
                                     n_excluded=0))
            continue
        sel = np.isin(blocks, years)
        tasks.append(dict(end_year=end, years=years, values=values[sel], blocks=blocks[sel],
                          amax=table.loc[years, "annual_max"].to_numpy(float), Tr=Tr,
                          thresh=thresh, nboot=nboot, seed=seed, ci=ci, gev_method=gev_method))
    if n_jobs == 1:
        results = map(analyse_window, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=n_jobs)
        results = pool.map(analyse_window, tasks)
    excl = []
    for i, (end, *est) in enumerate(results, 1):
        r, e = summarise(end, Tr, *est, ci)
        rows += r
        excl += e
        if i % 25 == 0 or i == len(tasks):
            print(f"  windows done: {i}/{len(tasks)}", flush=True)
    if n_jobs != 1:
        pool.shutdown()
    summary = pd.DataFrame(rows).set_index(["end_year", "Tr", "model"]).sort_index()
    exclusions = pd.DataFrame(excl, columns=["end_year", "Tr", "model", "reason", "n", "frac"])
    return summary, exclusions


def wide(summary, T):
    """One row per window for return period T (columns gev_*, mev_*, diff_*)."""
    s = summary.xs(float(T), level="Tr")
    out = {}
    for model, tag in (("GEV", "gev"), ("MEV", "mev"), ("MEV-GEV", "diff")):
        m = s.xs(model, level="model")
        out[f"{tag}_rl"], out[f"{tag}_ci_l"], out[f"{tag}_ci_u"] = m["estimate"], m["ci_l"], m["ci_u"]
        out[f"{tag}_status"], out[f"{tag}_ci_status"] = m["point_status"], m["ci_status"]
        out[f"{tag}_n_valid"] = m["n_valid"]
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def _style(plt):
    plt.rcParams.update({"figure.figsize": (5, 3), "figure.dpi": 200, "font.size": 8,
                         "axes.linewidth": 0.5, "lines.linewidth": 1.0, "legend.frameon": False,
                         "xtick.direction": "in", "ytick.direction": "in",
                         "savefig.bbox": "tight", "savefig.pad_inches": 0.05})


def plot_rolling(w, T, path, gev_method):
    """Return levels per window with bootstrap bands; withheld values show as gaps."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for tag, model, label in (("gev", "GEV", GEV_LABEL[gev_method]), ("mev", "MEV", "MEV")):
        ax.plot(w.index, w[f"{tag}_rl"], label=label, color=COLORS[model])
        ax.fill_between(w.index, w[f"{tag}_ci_l"], w[f"{tag}_ci_u"], color=COLORS[model], alpha=0.2, lw=0)
    ax.set_xlabel("Year (end of window)")
    ax.set_ylabel(f"{T:g}-year daily rainfall depth [mm]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


def plot_return_levels(model, gev_fit, amax, boot, path, gev_method):
    import matplotlib.pyplot as plt
    T = np.logspace(np.log10(1.1), 3, 200)
    fig, ax = plt.subplots()
    ax.plot(T, model.return_level(T), color=COLORS["MEV"], label="MEV")
    ax.plot(T, gev_return_levels(gev_fit, T)[0], color=COLORS["GEV"], label=GEV_LABEL[gev_method])
    ax.errorbar(boot.return_periods, boot.estimate,
                yerr=[boot.estimate - boot.lower, boot.upper - boot.estimate],
                fmt="none", ecolor=COLORS["MEV"], capsize=2, lw=0.8,
                label=f"MEV {boot.ci_level:.0%} bootstrap CI")
    x = np.sort(amax)
    ax.plot(1 / (1 - np.arange(1, x.size + 1) / (x.size + 1)), x, "k.", ms=3,
            label="annual maxima (Weibull plotting position)")
    ax.set_xscale("log")
    ax.set_xlabel("Return period [years]")
    ax.set_ylabel("Daily rainfall depth [mm]")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend()
    fig.savefig(path)
    plt.close(fig)


# --------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description="MEVD rainfall example and MEV vs GEV sliding windows")
    p.add_argument("--data", default="data/pd.csv", help="wide day-by-year CSV (default: data/pd.csv)")
    p.add_argument("--outdir", default="mevpy_output")
    p.add_argument("--thresh", type=float, default=1.0, help="ordinary-event threshold [mm]")
    p.add_argument("--min-completeness", type=float, default=0.8)
    p.add_argument("--ws", type=int, default=30, help="window length [years]")
    p.add_argument("--min-years", type=int, default=20)
    p.add_argument("--tr", type=float, nargs="+", default=[100.0])
    p.add_argument("--gev-method", choices=("lmom", "mle"), default="lmom",
                   help="GEV estimator: L-moments (default) or maximum likelihood")
    p.add_argument("--nboot", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ci-level", type=float, default=0.90)
    p.add_argument("--min-valid-count", type=int, default=10)
    p.add_argument("--min-valid-frac", type=float, default=0.90)
    p.add_argument("--n-jobs", type=int, default=os.cpu_count() or 1)
    p.add_argument("--skip-rolling", action="store_true", help="only the single-window demo")
    a = p.parse_args(argv)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    ci = dict(ci_level=a.ci_level, min_valid_count=a.min_valid_count, min_valid_frac=a.min_valid_frac)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _style(plt)

    # 1. Load and prepare ------------------------------------------------
    daily = rainfall.read_day_by_year_csv(a.data)
    table = rainfall.yearly_table(daily, a.thresh, a.min_completeness)
    table.to_csv(out / "year_table.csv")
    print(f"{len(table)} years {table.index.min()}-{table.index.max()}, {table['valid'].sum()} valid, "
          f"{(table['valid'] & (table['n_events'] == 0)).sum()} valid dry years")

    # 2. Fit the MEVD on the latest window ---------------------------------
    last = table.index.max()
    years = table.index[table["valid"] & (table.index > last - a.ws)].to_numpy()
    values, blocks = rainfall.ordinary_daily_events(daily, a.thresh, years)
    model = MEVD.fit(values, blocks, WeibullPWM(loc=a.thresh), block_index=years)
    model.to_frame().to_csv(out / "demo_parameters.csv", index=False)
    print(f"\nWindow {years.min()}-{years.max()}: {model}")
    print(model.to_frame().head().to_string(index=False))

    # 3. CDF and quantiles ---------------------------------------------------
    x = np.array([50.0, 100.0, 150.0])
    print("\nH(x) for x =", x, "mm:", np.round(model.cdf(x), 5))
    q = np.array([0.5, 0.9, 0.99])
    print("ppf(q) for q =", q, ":", np.round(model.ppf(q), 2), "mm;  median", round(model.median(), 2))

    # 4. Return levels with bootstrap intervals -----------------------------
    T_demo = np.array([2, 5, 10, 20, 50, 100, 200])
    boot = bootstrap(model, T_demo, n_boot=a.nboot, random_state=a.seed, **ci)
    s = boot.summary()
    amax = table.loc[years, "annual_max"].to_numpy(float)
    gev_fit = fit_gev(amax, a.gev_method)
    s["gev"] = gev_return_levels(gev_fit, T_demo)[0]
    s.to_csv(out / "demo_return_levels.csv")
    print(f"\nReturn levels [mm], {a.ci_level:.0%} bootstrap CI ({a.nboot} replicates); "
          f"gev = {GEV_LABEL[a.gev_method]}:")
    print(s.round(2).to_string())
    print(f"failed replicate estimates: {len(boot.failures())}")
    plot_return_levels(model, gev_fit, amax, boot, out / "demo_return_levels.pdf", a.gev_method)

    # 5. Sliding-window MEV vs GEV --------------------------------------------
    if not a.skip_rolling:
        Tr = np.asarray(a.tr, dtype=float)
        print(f"\nSliding windows: ws={a.ws}, min_years={a.min_years}, Tr={Tr.tolist()}, "
              f"GEV={a.gev_method}, nboot={a.nboot}, n_jobs={a.n_jobs}")
        summary, exclusions = rolling_comparison(daily, table, a.ws, a.min_years, a.thresh, Tr,
                                                 a.nboot, a.seed, ci, a.n_jobs, a.gev_method)
        summary.to_csv(out / "summary.csv")
        exclusions.to_csv(out / "exclusions.csv", index=False)
        for T in Tr:
            w = wide(summary, T)
            w.to_csv(out / f"rolling_Tr{T:g}.csv")
            plot_rolling(w, T, out / f"rolling_Tr{T:g}.pdf", a.gev_method)
        print("Excluded bootstrap estimates (all windows):")
        print(summary.groupby(["Tr", "model"])[["n_boot", "n_excluded"]].sum().to_string())
    print(f"\nResults written to {out.resolve()}")


if __name__ == "__main__":
    main()
