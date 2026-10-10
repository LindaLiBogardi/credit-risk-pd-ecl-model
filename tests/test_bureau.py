"""Unit tests for the bureau aggregation."""

import pandas as pd

from src.bureau import aggregate_bureau


def test_aggregate_bureau():
    bureau = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "SK_ID_BUREAU": [10, 11, 12],
        "CREDIT_ACTIVE": ["Active", "Closed", "Sold"],
        "CREDIT_TYPE": ["Consumer credit", "Microloan", "Credit card"],
        "AMT_CREDIT_SUM_DEBT": [100.0, 50.0, 0.0],
        "CREDIT_DAY_OVERDUE": [0, 0, 30],
        "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 20.0],
        "DAYS_CREDIT": [-100, -500, -50],
    })
    result = aggregate_bureau(bureau).set_index("SK_ID_CURR")

    assert result.loc[1, "BUREAU_LOAN_COUNT"] == 2
    assert result.loc[1, "BUREAU_ACTIVE_COUNT"] == 1
    assert result.loc[1, "BUREAU_ACTIVE_DEBT"] == 100.0      # closed loan's debt excluded
    assert result.loc[1, "BUREAU_DAYS_SINCE_LAST_CREDIT"] == -100
    assert result.loc[1, "BUREAU_ANY_DELINQUENCY"] == 1      # microloan
    assert result.loc[2, "BUREAU_SOLD_OR_BAD_COUNT"] == 1
    assert result.loc[2, "BUREAU_ANY_DELINQUENCY"] == 1      # overdue loan