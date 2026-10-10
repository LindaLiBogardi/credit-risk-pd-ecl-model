"""Derived features for the Home Credit application data.

Adds four ratios: CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_TERM
and CREDIT_GOODS_RATIO. AGE_YEARS is derived from DAYS_BIRTH.
"""

import numpy as np
import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add affordability and loan structure ratios, plus AGE_YEARS.

    Infinite values, from division by zero, are replaced with NaN so that
    WoE binning treats them as missing.
    """
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