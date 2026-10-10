"""Scenario adjustments of PD for stress testing."""

import numpy as np
import pandas as pd
from scipy.optimize import brentq


def shift_pd(pd_: pd.Series, delta: float) -> pd.Series:
    """Shift PD on the log-odds scale by delta."""
    logit = np.log(pd_ / (1 - pd_))
    return 1 / (1 + np.exp(-(logit + delta)))


def calibrate_delta(pd_: pd.Series, target_mean_pd: float) -> float:
    """Find the log-odds shift that gives the target average PD."""
    return brentq(lambda d: shift_pd(pd_, d).mean() - target_mean_pd, -5, 5)