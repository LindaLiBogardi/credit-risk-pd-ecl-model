"""Unit tests for the data cleaning functions."""

import pandas as pd

from src.data_cleaning import drop_excluded_columns, fix_days_employed


def test_fix_days_employed_replaces_placeholder():
    df = pd.DataFrame({"DAYS_EMPLOYED": [-1000, 365243]})
    result = fix_days_employed(df)
    assert result["DAYS_EMPLOYED"].isna().tolist() == [False, True]
    assert result["DAYS_EMPLOYED_ANOM"].tolist() == [0, 1]


def test_fix_days_employed_keeps_input_unchanged():
    df = pd.DataFrame({"DAYS_EMPLOYED": [365243]})
    fix_days_employed(df)
    assert df["DAYS_EMPLOYED"].iloc[0] == 365243


def test_drop_excluded_columns():
    df = pd.DataFrame(columns=["TARGET", "CODE_GENDER", "WEEKDAY_APPR_PROCESS_START",
                               "NAME_TYPE_SUITE", "APARTMENTS_AVG", "FLOORSMAX_MODE"])
    result = drop_excluded_columns(df)
    assert list(result.columns) == ["TARGET"]