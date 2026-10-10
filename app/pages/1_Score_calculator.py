"""Score calculator: choose a client's characteristics and see the score and PD."""

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "data"

LABELS = {
    "EXT_SOURCE_3": "External credit score 3",
    "EXT_SOURCE_2": "External credit score 2",
    "EXT_SOURCE_1": "External credit score 1",
    "DAYS_EMPLOYED": "Employment length (days, negative)",
    "OCCUPATION_TYPE": "Occupation",
    "ORGANIZATION_TYPE": "Employer industry",
    "NAME_EDUCATION_TYPE": "Education",
    "REGION_RATING_CLIENT_W_CITY": "Region rating (1 = best)",
    "AMT_GOODS_PRICE": "Price of goods financed",
    "CREDIT_GOODS_RATIO": "Loan amount / goods price",
    "CREDIT_TERM": "Number of payments",
    "OWN_CAR_AGE": "Age of own car (years)",
}


@st.cache_data
def load():
    scorecard = pd.read_csv(DATA / "scorecard.csv")
    scorecard = scorecard[scorecard["bin"] != "Special"]   # no clients in these bins
    scaling = json.loads((DATA / "metrics.json").read_text())["scaling"]
    return scorecard, scaling


def score_to_pd(score: float, scaling: dict) -> float:
    """Invert the scorecard scaling: score -> odds of being good -> PD."""
    factor = scaling["pdo"] / np.log(2)
    offset = scaling["base_score"] - factor * np.log(scaling["base_odds"])
    odds_good = np.exp((score - offset) / factor)
    return 1 / (1 + odds_good)


def short(label: str, width: int = 70) -> str:
    return label if len(label) <= width else label[:width - 1] + "…"


st.set_page_config(page_title="Score calculator", layout="wide")
st.title("Score calculator")
st.write(
    "Choose a client's characteristics. Each answer gives points; the total score maps "
    "to a probability of default. Higher scores mean lower risk."
)

scorecard, scaling = load()
left, right = st.columns([2, 3])

chosen = []
with left:
    for var, label in LABELS.items():
        options = scorecard[scorecard["variable"] == var].reset_index(drop=True)
        idx = st.selectbox(
            label,
            options.index,
            index=len(options) // 2,
            format_func=lambda i, o=options: short(str(o.loc[i, "bin"])),
            key=var,
        )
        chosen.append({"Variable": label, "Points": int(options.loc[idx, "points"])})

breakdown = pd.DataFrame(chosen)
total = int(breakdown["Points"].sum())
pd_value = score_to_pd(total, scaling)

with right:
    st.metric("Total score", total)
    st.metric("Probability of default (12 months)", f"{pd_value:.1%}")
    st.caption(
        f"Scaling: {scaling['base_score']} points at good:bad odds of "
        f"{scaling['base_odds']}:1; every {scaling['pdo']} points doubles the odds."
    )
    st.subheader("Points by variable")
    chart = (
        alt.Chart(breakdown)
        .mark_bar()
        .encode(
            x=alt.X("Points:Q", title="Points"),
            y=alt.Y("Variable:N", sort="-x", title=None,
                    axis=alt.Axis(labelLimit=400)),
            tooltip=["Variable", "Points"],
        )
        .properties(height=380)
    )
    st.altair_chart(chart, use_container_width=True)