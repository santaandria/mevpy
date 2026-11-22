"""mevpy -- empirical Metastatistical Extreme Value Distribution.

    H(x) = (1 / M) * sum_j F(x; theta_j) ** n_j

Quick start::

    from mevpy import MEVD, WeibullPWM, bootstrap
    model = MEVD.fit(depths, years, distribution=WeibullPWM(loc=1.0), block_index=valid_years)
    model.cdf(100.0); model.ppf([0.5, 0.99]); model.return_level([10, 100])
    ci = bootstrap(model, [10, 100], n_boot=1000, random_state=0)
"""
from .bootstrap import BootstrapResult, bootstrap, percentile_interval
from .core import MEVD, FitError, InversionError
from .ordinary import OrdinaryDistribution, ScipyOrdinary
from .weibull import WeibullPWM

__all__ = ["MEVD", "FitError", "InversionError", "OrdinaryDistribution", "ScipyOrdinary",
           "WeibullPWM", "bootstrap", "BootstrapResult", "percentile_interval"]
__version__ = "2.0.0"
