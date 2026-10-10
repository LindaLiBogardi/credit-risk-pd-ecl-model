"""ECL and stress test: probability-weighted ECL with adjustable LGD and scenario weights."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parent.parent / "data"
SCENARIO_ORDER = ["Base", "Adverse", "Severe"]


@st.cache_data
def load():
    ecl = pd.read_csv(DATA / "ecl_by_scenario_stage.csv")
    deciles = pd.read_csv(DATA / "ecl_by_decile.csv")
    metrics = json.loads((DATA / "metrics.json").read_text())
    return ecl, deciles, metrics


def billions(x: float) -> str:
    return f"{x / 1e9:,.2f} bn"


st.set_page_config(page_title="ECL and stress test", layout="wide")
st.title("ECL and stress test")
st.write(
    "IFRS 9 expected credit loss under three macro scenarios. Loans are re-staged in each "
    "scenario: a 12-month PD above 20% moves a loan to Stage 2, with lifetime ECL. "
    "Adjust the LGD and the scenario weights to see the effect on the provision."
)

ecl, deciles, metrics = load()
base_lgd = metrics["lgd"]

# Controls
with st.sidebar:
    st.header("Assumptions")
    lgd = st.slider("Loss given default (LGD)", 0.30, 0.90, base_lgd, 0.05,
                    format="%.2f")
    w_adverse = st.slider("Weight: adverse scenario", 0.0, 1.0, 0.30, 0.05)
    w_severe = st.slider("Weight: severe scenario", 0.0, 1.0, 0.10, 0.05)

w_base = 1.0 - w_adverse - w_severe
if w_base < 0:
    st.error("The adverse and severe weights add up to more than 100%. Lower one of them.")
    st.stop()
weights = {"Base": w_base, "Adverse": w_adverse, "Severe": w_severe}

# ECL is proportional to LGD, so all results scale by lgd / base_lgd
ecl["ecl_adj"] = ecl["ecl"] * lgd / base_lgd
by_scenario = ecl.groupby("scenario")["ecl_adj"].sum().reindex(SCENARIO_ORDER)
weighted = sum(by_scenario[s] * w for s, w in weights.items())
base_ecl = by_scenario["Base"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Probability-weighted ECL", billions(weighted))
c2.metric("Coverage (ECL / exposure)", f"{weighted / metrics['total_ead']:.2%}")
c3.metric("vs base scenario", f"{weighted / base_ecl - 1:+.1%}")
c4.metric("Base weight", f"{w_base:.0%}")

left, right = st.columns(2)

with left:
    st.subheader("ECL by scenario and stage")
    chart_data = ecl.assign(Stage="Stage " + ecl["stage"].astype(str))
    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("scenario:N", sort=SCENARIO_ORDER, title=None),
            y=alt.Y("ecl_adj:Q", title="ECL"),
            color=alt.Color("Stage:N", legend=alt.Legend(orient="bottom")),
            tooltip=["scenario", "Stage", "loans",
                     alt.Tooltip("ecl_adj:Q", format=",.0f", title="ECL")],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)

    stage2 = (
        ecl[ecl["stage"] == 2].set_index("scenario")["loans"]
        / ecl.groupby("scenario")["loans"].sum()
    ).reindex(SCENARIO_ORDER)
    st.caption(
        "Share of loans in Stage 2: "
        + ", ".join(f"{s} {stage2[s]:.1%}" for s in SCENARIO_ORDER)
        + ". Stage migration makes ECL rise faster than PD."
    )

with right:
    st.subheader("Where the expected loss comes from")
    deciles["ecl_share_pct"] = deciles["ecl_share"] * 100
    chart = (
        alt.Chart(deciles)
        .mark_bar()
        .encode(
            x=alt.X("PD_DECILE:O", title="PD decile (1 = safest)"),
            y=alt.Y("ecl_share_pct:Q", title="Share of 12-month ECL (%)"),
            tooltip=[alt.Tooltip("PD_DECILE:O", title="Decile"),
                     alt.Tooltip("ecl_share_pct:Q", format=".1f", title="ECL share %")],
        )
        .properties(height=360)
    )
    st.altair_chart(chart, use_container_width=True)
    st.caption("Base scenario, LGD 45%. The riskiest decile holds 8% of the exposure "
               "but 28% of the expected loss.")

st.info(
    "Assumptions: scenario PDs are derived by an assumed log-odds shift (no macro model "
    "was available), LGD is a fixed assumption since the data contains no recovery "
    "information, and the SICR rule is a 20% PD proxy."
)
