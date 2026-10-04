"""Derived features for the Home Credit application data."""
"""Four ratios are added to the dataset: CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_TERM, and CREDIT_GOODS_RATIO. Additionally, the AGE_YEARS feature is derived from DAYS_BIRTH."""

import numpy as np
import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add affordability and loan structure ratios."""
    df = df.copy()
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25

    ratio_cols = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
                  "CREDIT_TERM", "CREDIT_GOODS_RATIO"]
    df[ratio_cols] = df[ratio_cols].replace([np.inf, -np.inf], np.nan)
    return df