"""Expected credit loss (ECL) calculation under IFRS 9."""

import pandas as pd

LGD_UNSECURED = 0.45   # assumption: no recovery data available


def compute_ecl(pd_: pd.Series, ead: pd.Series, lgd: float = LGD_UNSECURED) -> pd.Series:
    """12-month expected credit loss per loan: ECL = PD x LGD x EAD."""
    return pd_ * lgd * ead