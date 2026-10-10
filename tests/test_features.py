"""Unit tests for the derived features."""

import numpy as np
import pandas as pd

from src.features import add_derived_features


def make_row(**overrides):
    row = {"AMT_CREDIT": 200.0, "AMT_INCOME_TOTAL": 100.0, "AMT_ANNUITY": 10.0,
           "AMT_GOODS_PRICE": 160.0, "DAYS_BIRTH": -36525}
    row.update(overrides)
    return pd.DataFrame([row])


def test_ratios_and_age():
    result = add_derived_features(make_row()).iloc[0]
    assert np.isclose(result["CREDIT_INCOME_RATIO"], 2.0)
    assert np.isclose(result["ANNUITY_INCOME_RATIO"], 0.1)
    assert np.isclose(result["CREDIT_TERM"], 20.0)
    assert np.isclose(result["CREDIT_GOODS_RATIO"], 1.25)
    assert np.isclose(result["AGE_YEARS"], 100.0)


def test_division_by_zero_gives_nan():
    result = add_derived_features(make_row(AMT_ANNUITY=0.0)).iloc[0]
    assert np.isnan(result["CREDIT_TERM"])