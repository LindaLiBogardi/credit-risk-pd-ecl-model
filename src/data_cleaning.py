import numpy as np
import pandas as pd

DAYS_EMPLOYED_PLACEHOLDER = 365243

EXCLUDED_COLUMNS = [
    "CODE_GENDER",                  # anti-discrimination
    "WEEKDAY_APPR_PROCESS_START",   # no business logic
    "NAME_TYPE_SUITE",              # no clear link to repayment
]
BUILDING_SUFFIXES = ("_AVG", "_MODE", "_MEDI")


def fix_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the DAYS_EMPLOYED placeholder with NaN and add a flag."""
    df = df.copy()
    is_placeholder = df["DAYS_EMPLOYED"] == DAYS_EMPLOYED_PLACEHOLDER
    df["DAYS_EMPLOYED_ANOM"] = is_placeholder.astype(int)
    df.loc[is_placeholder, "DAYS_EMPLOYED"] = np.nan
    return df


def drop_excluded_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop variables excluded up front in the expert long list."""
    building_cols = [c for c in df.columns if c.endswith(BUILDING_SUFFIXES)]
    return df.drop(columns=EXCLUDED_COLUMNS + building_cols)

