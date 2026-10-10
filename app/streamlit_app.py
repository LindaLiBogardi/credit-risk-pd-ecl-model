"""Credit risk PD and IFRS 9 ECL dashboard: overview page."""

import json
from pathlib import Path

import streamlit as st

DATA = Path(__file__).resolve().parent / "data"

st.set_page_config(page_title="Credit Risk PD & ECL", layout="wide")
metrics = json.loads((DATA / "metrics.json").read_text())

st.title("Credit Risk PD Model and IFRS 9 ECL")
st.write(
    "A probability of default scorecard and IFRS 9 expected credit loss calculation, "
    "built on the Home Credit Default Risk dataset. Use the pages in the sidebar to try "
    "the scorecard, explore the stress scenarios and review model quality and fairness."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Gini", f"{metrics['test_gini']:.3f}")
col2.metric("Scorecard variables", metrics["n_variables"])
col3.metric("Loans in portfolio", f"{metrics['loans']:,}")
col4.metric("LGD assumption", f"{metrics['lgd']:.0%}")

st.divider()
st.markdown(
    """
**How it was built**

1. **Data:** 307,511 loan applications, cleaned, with a stratified 70/30 train/test split.
2. **Variable selection:** expert long list, WoE/IV screening, correlation checks and
   backward elimination, from 122 columns to 12 variables.
3. **PD model:** logistic regression scorecard, 600 points at odds 50:1, 20 points to
   double the odds.
4. **IFRS 9:** 12-month and lifetime ECL, staging with a SICR proxy, and three
   probability-weighted macro scenarios.
5. **Fairness:** proxy test, 80% rule and calibration by gender and age group.
"""
)

