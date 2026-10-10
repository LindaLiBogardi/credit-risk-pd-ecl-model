"""Unit tests for the ECL functions."""

import numpy as np
import pandas as pd

from src.ecl import assign_stage, compute_ecl, lifetime_pd


def test_compute_ecl():
    result = compute_ecl(pd.Series([0.10]), pd.Series([1000.0]), lgd=0.5)
    assert result.iloc[0] == 50.0


def test_lifetime_pd_two_years():
    result = lifetime_pd(pd.Series([0.20]), pd.Series([2.0]))
    assert np.isclose(result.iloc[0], 0.36)


def test_lifetime_pd_at_least_one_year():
    result = lifetime_pd(pd.Series([0.20]), pd.Series([0.5]))
    assert np.isclose(result.iloc[0], 0.20)


def test_assign_stage():
    stages = assign_stage(pd.Series([0.05, 0.25]), threshold=0.20)
    assert list(stages) == [1, 2]