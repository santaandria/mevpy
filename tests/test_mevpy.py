"""Focused tests for mevpy (stdlib unittest; also collected by pytest).

Run from mevpy/:  python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mevpy import (MEVD, FitError, InversionError, OrdinaryDistribution, ScipyOrdinary,  # noqa: E402
                  WeibullPWM, bootstrap)


def synthetic(n_blocks=25, seed=1, loc=1.0, dry_blocks=()):
    """Weibull events with block-varying (n, C, W); some blocks may be dry."""
    rng = np.random.default_rng(seed)
    values, labels = [], []
    for j in range(n_blocks):
        if j in dry_blocks:
            continue
        n = rng.integers(30, 90)
        c, w = rng.uniform(6, 10), rng.uniform(0.7, 0.95)
        values.append(loc + c * rng.weibull(w, n))
        labels.append(np.full(n, 2000 + j))
    return np.concatenate(values), np.concatenate(labels), 2000 + np.arange(n_blocks)


class Exponential(OrdinaryDistribution):
    """Minimal custom ordinary distribution: exponential, theta = (scale,), MLE."""
    name, param_names, min_events = "expon", ("scale",), 1

    def fit(self, sample):
        return (float(np.mean(sample)),)

    def cdf(self, x, scale):
        return -np.expm1(-np.maximum(x, 0) / scale)

    def ppf(self, p, scale):
        return -scale * np.log1p(-p)

    def support(self, scale):
        return np.zeros_like(scale), np.full_like(scale, np.inf)


class TestDistribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v, b, idx = synthetic()
        cls.model = MEVD.fit(v, b, WeibullPWM(loc=1.0), block_index=idx)

    def test_cdf_bounds_and_monotone(self):
        x = np.linspace(-10, 2000, 5001)
        h = self.model.cdf(x)
        self.assertTrue(np.all((h >= 0) & (h <= 1)))
        self.assertTrue(np.all(np.diff(h) >= 0))
        self.assertAlmostEqual(self.model.cdf(1e9), 1.0)
        # at/below the lower support (loc = 1) H is 0
        self.assertEqual(self.model.cdf(0.5), 0.0)
        np.testing.assert_allclose(self.model.sf(x), 1 - h, atol=1e-12)

    def test_ppf_inverts_cdf(self):
        q = np.array([0.05, 0.2, 0.5, 0.9, 0.99, 0.999, 1 - 1e-8])
        np.testing.assert_allclose(self.model.cdf(self.model.ppf(q)), q, rtol=1e-9)
        x = np.array([30.0, 40.0, 120.0, 300.0])   # where H is not numerically flat
        np.testing.assert_allclose(self.model.ppf(self.model.cdf(x)), x, rtol=1e-8)

    def test_isf_tiny_probabilities(self):
        s = np.array([1e-4, 1e-8, 1e-12])
        np.testing.assert_allclose(self.model.sf(self.model.isf(s)), s, rtol=1e-9)

    def test_return_level_definition(self):
        T = np.array([2.0, 10.0, 100.0, 1000.0])
        rl = self.model.return_level(T)
        np.testing.assert_allclose(self.model.cdf(rl), 1 - 1 / T, rtol=1e-12)
        self.assertTrue(np.all(np.diff(rl) > 0))

    def test_endpoints(self):
        m = self.model
        self.assertEqual(m.ppf(0.0), 1.0)                     # lower support = loc
        self.assertEqual(m.ppf(1.0), np.inf)
        self.assertTrue(np.isnan(m.ppf(1.5)) and np.isnan(m.ppf(-0.1)) and np.isnan(m.ppf(np.nan)))
        self.assertEqual(m.support(), (1.0, np.inf))

    def test_scalar_and_array_shapes(self):
        m = self.model
        self.assertEqual(np.ndim(m.cdf(50.0)), 0)
        self.assertEqual(np.ndim(m.ppf(0.5)), 0)
        self.assertEqual(m.cdf(np.ones((2, 3))).shape, (2, 3))
        self.assertEqual(m.ppf(np.full((2, 2), 0.7)).shape, (2, 2))
        self.assertEqual(m.return_level([10, 100]).shape, (2,))
        self.assertEqual(m.ppf(0.7), m.ppf([0.7])[0])

    def test_pairing_is_preserved(self):
        f = self.model.to_frame()
        self.assertEqual(list(f.columns), ["block", "n", "C", "W"])
        self.assertTrue((f["n"].to_numpy() == self.model.n).all() and (f["n"] > 0).all())
        np.testing.assert_array_equal(f[["C", "W"]].to_numpy(), self.model.params)


class TestSingleBlock(unittest.TestCase):
    def test_identity_h_equals_f_power_n(self):
        rng = np.random.default_rng(0)
        x = 1.0 + 8 * rng.weibull(0.8, 60)
        dist = WeibullPWM(loc=1.0)
        m = MEVD.fit(x, np.zeros(60, int), dist)
        c, w = m.params[0]
        grid = np.linspace(1.0, 400, 400)
        np.testing.assert_allclose(m.cdf(grid), dist.cdf(grid, c, w) ** 60, rtol=1e-12, atol=1e-300)
        q = np.array([0.1, 0.5, 0.99])
        expected = dist.ppf(q ** (1 / 60), c, w)              # closed-form inverse of F^n
        np.testing.assert_allclose(m.ppf(q), expected, rtol=1e-10)

    def test_weibull_pwm_matches_greenwood_formula(self):
        # Greenwood (1979) PWM estimator written out term by term
        from scipy.special import gamma
        x = np.sort(np.random.default_rng(3).weibull(0.8, 50) * 7)
        n = x.size
        m0 = x.mean()
        m1 = sum(x[i] * (n - (i + 1)) for i in range(n)) / (n * (n - 1))
        c_ref = m0 / gamma(np.log(m0 / m1) / np.log(2))
        w_ref = np.log(2) / np.log(m0 / (2 * m1))
        np.testing.assert_allclose(WeibullPWM().fit(x), (c_ref, w_ref), rtol=1e-13)


class TestCustomDistribution(unittest.TestCase):
    def test_custom_subclass(self):
        rng = np.random.default_rng(5)
        n, scales = np.array([20, 50, 35]), np.array([5.0, 8.0, 6.0])
        v = np.concatenate([rng.exponential(s, k) for s, k in zip(scales, n)])
        b = np.repeat([0, 1, 2], n)
        m = MEVD.fit(v, b, Exponential())
        x = np.array([10.0, 30.0, 60.0])
        sc = m.params[:, 0]
        expected = np.mean([(1 - np.exp(-x / s)) ** k for s, k in zip(sc, n)], axis=0)
        np.testing.assert_allclose(m.cdf(x), expected, rtol=1e-12)
        q = np.array([0.3, 0.9, 0.999])
        np.testing.assert_allclose(m.cdf(m.ppf(q)), q, rtol=1e-9)   # default logcdf/isf used
        r = bootstrap(m, [10, 50], n_boot=30, random_state=1)
        self.assertTrue(r.valid.all())

    def test_scipy_adapter_gamma(self):
        rng = np.random.default_rng(6)
        v = np.concatenate([rng.gamma(0.8, 9.0, 60), rng.gamma(0.9, 7.0, 40)])
        b = np.repeat(["a", "b"], [60, 40])
        dist = ScipyOrdinary(stats.gamma, floc=0)
        m = MEVD.fit(v, b, dist)
        self.assertEqual(dist.param_names, ("a", "loc", "scale"))
        a0, l0, s0 = m.params[0]
        a1, l1, s1 = m.params[1]
        x = 80.0
        h = 0.5 * (stats.gamma.cdf(x, a0, l0, s0) ** 60 + stats.gamma.cdf(x, a1, l1, s1) ** 40)
        self.assertAlmostEqual(m.cdf(x), h, places=12)
        self.assertAlmostEqual(m.cdf(m.return_level(100)), 0.99, places=10)


class TestBootstrap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        v, b, idx = synthetic(seed=2)
        cls.model = MEVD.fit(v, b, WeibullPWM(1.0), block_index=idx)

    def test_reproducible(self):
        r1 = bootstrap(self.model, [10, 100], n_boot=60, random_state=42)
        r2 = bootstrap(self.model, [10, 100], n_boot=60, random_state=np.random.default_rng(42))
        np.testing.assert_array_equal(r1.indices, r2.indices)
        np.testing.assert_array_equal(r1.replicates, r2.replicates)
        np.testing.assert_array_equal(r1.lower, r2.lower)
        r3 = bootstrap(self.model, [10, 100], n_boot=60, random_state=43)
        self.assertFalse(np.array_equal(r1.indices, r3.indices))

    def test_resampling_unit_and_interval(self):
        r = bootstrap(self.model, [100], n_boot=200, ci_level=0.8, random_state=0)
        self.assertEqual(r.indices.shape, (200, self.model.n_blocks))
        # replicate 0 is exactly the MEVD of the drawn (n_j, theta_j) pairs
        rows = r.indices[0]
        direct = MEVD(self.model.distribution, self.model.blocks[rows], self.model.n[rows],
                      self.model.params[rows]).return_level(100)
        self.assertEqual(r.replicates[0, 0], direct)
        lo, hi = np.percentile(r.replicates[:, 0], [10, 90])
        self.assertEqual((r.lower[0], r.upper[0]), (lo, hi))
        self.assertTrue(r.lower[0] < r.estimate[0] < r.upper[0])
        self.assertEqual(r.ci_status[0], "ok")

    def test_failures_reported_not_dropped(self):
        class Failing(MEVD):
            calls = 0

            def _quantile(self, s, q):
                Failing.calls += 1
                if Failing.calls % 4 == 0:
                    raise InversionError("forced")
                return super()._quantile(s, q)

        m = Failing(self.model.distribution, self.model.blocks, self.model.n, self.model.params)
        m._take = lambda idx: Failing(m.distribution, m.blocks[idx], m.n[idx], m.params[idx])
        r = bootstrap(m, [100], n_boot=40, random_state=0)
        failed = ~r.valid[:, 0]
        self.assertGreater(failed.sum(), 0)
        self.assertTrue(np.all(np.isnan(r.replicates[failed, 0])))
        self.assertEqual(len(r.failures()), failed.sum())
        self.assertTrue(r.failures()["status"].str.contains("forced").all())
        self.assertEqual(r.n_valid[0], 40 - failed.sum())
        # > 10 % failed -> interval withheld (min_valid_frac = 0.9)
        self.assertEqual(r.ci_status[0], "insufficient_valid_fraction")
        self.assertTrue(np.isnan(r.lower[0]))


class TestInvalidInput(unittest.TestCase):
    def setUp(self):
        self.v, self.b, self.idx = synthetic(n_blocks=5)

    def test_nan_values_rejected(self):
        v = self.v.copy()
        v[3] = np.nan
        with self.assertRaisesRegex(ValueError, "missing observations"):
            MEVD.fit(v, self.b, WeibullPWM(1.0))

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            MEVD.fit(self.v, self.b[:-1], WeibullPWM(1.0))

    def test_events_outside_block_index(self):
        with self.assertRaisesRegex(ValueError, "not listed"):
            MEVD.fit(self.v, self.b, WeibullPWM(1.0), block_index=self.idx[:-1])

    def test_duplicate_block_index(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            MEVD.fit(self.v, self.b, WeibullPWM(1.0), block_index=np.r_[self.idx, self.idx[0]])

    def test_too_few_events_in_block(self):
        v = np.r_[self.v, 5.0]
        b = np.r_[self.b, 9999]
        with self.assertRaisesRegex(FitError, "9999.*1 event"):
            MEVD.fit(v, b, WeibullPWM(1.0))

    def test_events_below_loc(self):
        with self.assertRaisesRegex(FitError, "> loc"):
            MEVD.fit(self.v, self.b, WeibullPWM(loc=5.0))

    def test_degenerate_block(self):
        with self.assertRaisesRegex(FitError, "invalid parameters"):
            MEVD.fit(np.full(10, 3.0), np.zeros(10), WeibullPWM(1.0))

    def test_all_blocks_dry(self):
        with self.assertRaises(FitError):
            MEVD.fit(np.array([]), np.array([], int), WeibullPWM(1.0), block_index=[1, 2])

    def test_bad_return_periods(self):
        m = MEVD.fit(self.v, self.b, WeibullPWM(1.0))
        for T in (1.0, 0.5, np.inf, np.nan, -3):
            with self.assertRaises(ValueError):
                m.return_level(T)
        with self.assertRaises(ValueError):
            bootstrap(m, [1.0], n_boot=5)
        with self.assertRaises(ValueError):
            bootstrap(m, [10], n_boot=5, ci_level=1.5)

    def test_bad_direct_construction(self):
        with self.assertRaises(ValueError):
            MEVD(WeibullPWM(), [1, 2], [3, -1], [[1, 1], [1, 1]])
        with self.assertRaises(ValueError):
            MEVD(WeibullPWM(), [1, 2], [3, 4], [[1, 1], [np.nan, 1]])


class TestRainfall(unittest.TestCase):
    def test_missing_years_excluded_not_dry(self):
        import pandas as pd
        from mevpy import rainfall
        idx = pd.date_range("2000-01-01", "2003-12-31", freq="D")
        rng = np.random.default_rng(0)
        s = pd.Series(np.where(rng.random(idx.size) < 0.3, rng.exponential(8, idx.size), 0.0), index=idx)
        s["2001"] = np.nan                                     # fully missing year
        s["2002-01-01":"2002-04-30"] = np.nan                  # 67 % observed
        s["2003"] = 0.0                                        # observed, dry
        t = rainfall.yearly_table(s, threshold=1.0)
        self.assertEqual(t["valid"].tolist(), [True, False, False, True])
        self.assertEqual(t.loc[2003, "n_events"], 0)
        years = t.index[t["valid"]].to_numpy()
        v, b = rainfall.ordinary_daily_events(s, 1.0, years)
        with self.assertRaisesRegex(FitError, "zero-event"):   # dry year rejected, not dropped
            MEVD.fit(v, b, WeibullPWM(1.0), block_index=years)
        m = MEVD.fit(v, b, WeibullPWM(1.0), block_index=years[:1])   # explicit exclusion
        self.assertEqual(m.blocks.tolist(), [2000])


class TestGEV(unittest.TestCase):
    def setUp(self):
        self.amax = stats.genextreme.rvs(-0.1, loc=60, scale=15, size=40, random_state=7)

    def test_lmom_reproduces_sample_lmoments(self):
        from scipy.special import gamma
        from mevpy.gev import _sample_lmoments, fit_gev_lmom
        fit = fit_gev_lmom(self.amax)
        self.assertEqual(fit.status, "ok")
        k, loc, scale = fit.params
        # GEV L-moments (Hosking 1990) with shape k = scipy's c
        l1 = loc + scale * (1 - gamma(1 + k)) / k
        l2 = scale * (1 - 2.0 ** -k) * gamma(1 + k) / k
        t3 = 2 * (1 - 3.0 ** -k) / (1 - 2.0 ** -k) - 3
        s1, s2, s3 = _sample_lmoments(self.amax)
        np.testing.assert_allclose([l1, l2, t3], [s1, s2, s3 / s2], rtol=1e-10)

    def test_default_is_lmom_and_mle_available(self):
        from mevpy.gev import fit_gev, fit_gev_lmom, fit_gev_mle, gev_return_levels
        self.assertEqual(fit_gev(self.amax).params, fit_gev_lmom(self.amax).params)
        mle = fit_gev(self.amax, "mle")
        self.assertEqual(mle.params, fit_gev_mle(self.amax).params)
        ref = stats.genextreme.fit(self.amax)                 # scipy's MLE is accepted here
        np.testing.assert_allclose(mle.params, ref, rtol=1e-6)
        rl, st = gev_return_levels(fit_gev(self.amax), [10, 100])
        self.assertTrue((st == "ok").all() and rl[1] > rl[0])
        with self.assertRaises(ValueError):
            fit_gev(self.amax, "moments")

    def test_invalid_samples_reported(self):
        from mevpy.gev import fit_gev, gev_return_levels
        for method in ("lmom", "mle"):
            self.assertEqual(fit_gev([1.0, 2.0], method).status, "gev_insufficient_data")
            self.assertNotEqual(fit_gev(np.full(10, 5.0), method).status, "ok")
            rl, st = gev_return_levels(fit_gev([1.0, np.nan, 3.0, 4.0], method), [100])
            self.assertTrue(np.isnan(rl[0]) and st[0] == "gev_insufficient_data")


if __name__ == "__main__":
    unittest.main()
