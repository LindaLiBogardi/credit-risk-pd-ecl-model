"""WoE transformation using fitted binners."""

import joblib
import pandas as pd


def load_binners(path):
    """Load the dictionary of fitted OptimalBinning objects."""
    return joblib.load(path)


def woe_transform(df: pd.DataFrame, binners: dict) -> pd.DataFrame:
    """Replace each variable with the WoE of its bin, using the fitted binners."""
    return pd.DataFrame(
        {var: b.transform(df[var], metric="woe") for var, b in binners.items()},
        index=df.index,
    )