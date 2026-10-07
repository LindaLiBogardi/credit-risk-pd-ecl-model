"""Expected credit loss (ECL) calculation under IFRS 9."""

import numpy as np
import pandas as pd

LGD_UNSECURED = 0.45        # assumption: no recovery data available
SICR_PD_THRESHOLD = 0.20    # proxy: no origination PD available


def compute_ecl(pd_: pd.Series, ead: pd.Series, lgd: float = LGD_UNSECURED) -> pd.Series:
    """12-month expected credit loss per loan: ECL = PD x LGD x EAD."""
    return pd_ * lgd * ead


def lifetime_pd(pd_12m: pd.Series, term_years: pd.Series) -> pd.Series:
    """Lifetime PD assuming a constant annual default rate over the remaining term."""
    years = np.maximum(term_years, 1.0)   # at least 12 months
    return 1 - (1 - pd_12m) ** years


def assign_stage(pd_12m: pd.Series, threshold: float = SICR_PD_THRESHOLD) -> pd.Series:
    """Stage 2 if the 12-month PD exceeds the SICR threshold, otherwise Stage 1."""
    return np.where(pd_12m > threshold, 2, 1)