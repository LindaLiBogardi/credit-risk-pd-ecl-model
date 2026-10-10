"""Model quality and fairness: discrimination, calibration and group outcomes."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from theme import BASELINE, RISK_OVER, RISK_UNDER, show

DATA = Path(__file__).resolve().parent.parent / "data"


@st.cache_data
def load():
    roc = pd.read_csv(DATA / "roc_curve.csv")
    calib = pd.read_csv(DATA / "calibration.csv")
    fairness = pd.read_csv(DATA / "fairness.csv")
    metrics = json.loads((DATA / "metrics.json").read_text())
    return roc, calib, fairness, metrics


def diagonal(max_value: float = 1.0) -> alt.Chart:
    line = pd.DataFrame({"x": [0, max_value], "y": [0, max_value]})
    return alt.Chart(line).mark_line(strokeDash=[4, 4], color=BASELINE).encode(
        x="x:Q", y="y:Q")


st.set_page_config(page_title="Model quality and fairness", layout="wide")
st.title("Model quality and fairness")

roc, calib, fairness, metrics = load()

st.header("Model quality (test set)")
st.write(
    f"The scorecard reaches a test Gini of **{metrics['test_gini']:.3f}** "
    f"(AUC {metrics['test_auc']:.3f}). Predicted and observed default rates agree "
    "closely in every PD decile."
)

left, right = st.columns(2)

with left:
    st.subheader("ROC curve")
    curve = alt.Chart(roc).mark_line(strokeWidth=2.5).encode(
        x=alt.X("fpr:Q", title="False positive rate"),
        y=alt.Y("tpr:Q", title="True positive rate"),
    )
    show((curve + diagonal()).properties(height=360))
    st.caption("The further the curve bulges above the dashed diagonal (a random model), "
               "the better the model separates defaulters from non-defaulters.")

with right:
    st.subheader("Calibration by PD decile")
    calib_pct = calib.assign(predicted=calib["predicted"] * 100,
                             observed=calib["observed"] * 100)
    upper = float(calib_pct[["predicted", "observed"]].max().max()) * 1.1
    points = alt.Chart(calib_pct).mark_line(point=True, strokeWidth=2.5).encode(
        x=alt.X("predicted:Q", title="Predicted PD (%)"),
        y=alt.Y("observed:Q", title="Observed default rate (%)"),
        tooltip=["decile",
                 alt.Tooltip("predicted:Q", format=".2f"),
                 alt.Tooltip("observed:Q", format=".2f")],
    )
    show((points + diagonal(upper)).properties(height=360))
    st.caption("Points on the dashed line mean the predicted PD matches reality.")

st.header("Fairness")
st.write(
    "Gender is not used in the model; it is joined back for testing only. The table shows "
    "outcomes by group if the bank approved the 80% of applicants with the highest scores."
)

table = fairness.assign(
    passes_80_rule=fairness["ratio_vs_highest"] >= 0.8,
    gap_pp=(fairness["observed_default"] - fairness["predicted_pd"]) * 100,
)
st.dataframe(
    table[["attribute", "group", "clients", "observed_default", "predicted_pd",
           "approval_rate", "ratio_vs_highest", "passes_80_rule"]],
    column_config={
        "attribute": "Attribute",
        "group": "Group",
        "clients": st.column_config.NumberColumn("Clients", format="%d"),
        "observed_default": st.column_config.NumberColumn("Observed default", format="percent"),
        "predicted_pd": st.column_config.NumberColumn("Predicted PD", format="percent"),
        "approval_rate": st.column_config.NumberColumn("Approval rate", format="percent"),
        "ratio_vs_highest": st.column_config.NumberColumn("Ratio vs highest", format="%.2f"),
        "passes_80_rule": "Passes 80% rule",
    },
    hide_index=True,
    use_container_width=True,
)

st.subheader("Calibration gap by group")
gap_chart = alt.Chart(table).mark_bar(cornerRadiusEnd=3).encode(
    x=alt.X("gap_pp:Q", title="Observed minus predicted (percentage points)"),
    y=alt.Y("group:N", title=None, sort=None),
    color=alt.condition(alt.datum.gap_pp > 0, alt.value(RISK_UNDER), alt.value(RISK_OVER)),
    tooltip=["attribute", "group", alt.Tooltip("gap_pp:Q", format="+.2f")],
).properties(height=220)
show(gap_chart)
st.caption(
    "Slate bars: the model over-predicts risk for the group (scored too harshly). "
    "Red bars: risk is under-predicted."
)

st.info(
    "Findings: gender passes the 80% rule, but clients under 30 fail it. The model "
    "over-predicts risk for women and young clients relative to their observed default "
    "rates. Correcting the gender gap would require using gender, which anti-discrimination "
    "rules prohibit: a known trade-off between fairness definitions. The 80% rule is a "
    "screening heuristic, not an EU legal standard."
)

