"""Credit risk PD and IFRS 9 ECL dashboard: overview page."""

import streamlit as st

st.set_page_config(page_title="Credit Risk PD & ECL", layout="wide")

st.title("Credit Risk PD Model and IFRS 9 ECL")
st.write(
    "A probability of default scorecard and IFRS 9 expected credit loss calculation, "
    "built on the Home Credit Default Risk dataset."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Test Gini", "0.490")
col2.metric("Variables", "12")
col3.metric("12-month ECL coverage", "3.24%")
col4.metric("Weighted ECL vs base", "+23%")