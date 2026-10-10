"""Shared Nordic-minimal styling for the dashboard charts.

All colours live here, so every page stays consistent.
"""

import altair as alt

ACCENT = "#1F5C4A"        # pine green: the single accent colour
ACCENT_LIGHT = "#A9C5BA"  # pale green: secondary series
TEXT = "#1F2933"
MUTED = "#6B7680"         # axis labels and captions
GRID = "#E6E8EA"          # light gridlines
BASELINE = "#B0B7BF"      # reference lines (random model, perfect calibration)
RISK_UNDER = "#B5473A"    # muted red: risk under-predicted
RISK_OVER = "#5B7A99"     # slate blue: risk over-predicted

STAGE_SCALE = alt.Scale(domain=["Stage 1", "Stage 2"], range=[ACCENT_LIGHT, ACCENT])


def style(chart: alt.Chart) -> alt.Chart:
    """Apply the shared look: quiet axes, light grid, no frame."""
    return (
        chart.configure(font="sans-serif")
        .configure_axis(
            grid=True, gridColor=GRID, domain=False, ticks=False,
            labelColor=MUTED, titleColor=MUTED,
            labelFontSize=12, titleFontSize=12, titleFontWeight="normal", labelPadding=6,
        )
        .configure_view(strokeWidth=0)
        .configure_legend(labelColor=MUTED, titleColor=MUTED, labelFontSize=12)
        .configure_mark(color=ACCENT)
    )


def show(chart: alt.Chart) -> None:
    """Render a chart with the shared style, bypassing Streamlit's default chart theme."""
    import streamlit as st

    st.altair_chart(style(chart), use_container_width=True, theme=None)

