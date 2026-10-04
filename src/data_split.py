"""Train/test split for the PD model."""

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.3


def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on TARGET with a fixed seed."""
    return train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["TARGET"],
        random_state=RANDOM_STATE,
    )