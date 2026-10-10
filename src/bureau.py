"""Aggregate credit bureau history to one row per client."""

import pandas as pd


def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Summarise each client's loans at other lenders into a few features."""
    b = bureau.copy()
    b["IS_ACTIVE"] = b["CREDIT_ACTIVE"] == "Active"
    b["IS_SOLD_OR_BAD"] = b["CREDIT_ACTIVE"].isin(["Sold", "Bad debt"])
    b["IS_MICROLOAN"] = b["CREDIT_TYPE"] == "Microloan"
    b["ACTIVE_DEBT"] = b["AMT_CREDIT_SUM_DEBT"].where(b["IS_ACTIVE"])

    agg = b.groupby("SK_ID_CURR").agg(
        BUREAU_LOAN_COUNT=("SK_ID_BUREAU", "count"),
        BUREAU_ACTIVE_COUNT=("IS_ACTIVE", "sum"),
        BUREAU_ACTIVE_DEBT=("ACTIVE_DEBT", "sum"),
        BUREAU_MAX_DAYS_OVERDUE=("CREDIT_DAY_OVERDUE", "max"),
        BUREAU_OVERDUE_AMOUNT=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        BUREAU_DAYS_SINCE_LAST_CREDIT=("DAYS_CREDIT", "max"),
        BUREAU_SOLD_OR_BAD_COUNT=("IS_SOLD_OR_BAD", "sum"),
        BUREAU_MICROLOAN_COUNT=("IS_MICROLOAN", "sum"),
    )
    agg["BUREAU_ANY_DELINQUENCY"] = (
        (agg["BUREAU_MAX_DAYS_OVERDUE"] > 0) | (agg["BUREAU_MICROLOAN_COUNT"] > 0)
    ).astype(int)
    return agg.reset_index()