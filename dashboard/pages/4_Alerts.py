"""High-confidence negative-signal page."""

from __future__ import annotations

import streamlit as st

from dashboard.components.page_shell import (
    configure_page,
    load_reviews,
)


configure_page("Alerts")
st.caption("Review high-confidence negative signals that may need attention.")

reviews = load_reviews()
threshold_percent = st.slider(
    "Minimum confidence",
    min_value=50,
    max_value=100,
    value=85,
    step=1,
    format="%d%%",
)
threshold = threshold_percent / 100

alerts = (
    reviews[
        reviews["sentiment"].eq("Negative")
        & reviews["confidence"].ge(threshold)
    ]
    .sort_values(
        ["confidence", "date"],
        ascending=False,
    )
)
alerts_display = alerts.copy()
alerts_display["confidence_pct"] = (
    alerts_display.pop("confidence") * 100
)

metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
metric_col_1.metric("Active signals", f"{len(alerts):,}")
metric_col_2.metric(
    "Brands affected",
    f"{alerts['brand'].nunique():,}",
)
metric_col_3.metric(
    "Topics affected",
    f"{alerts['topic'].nunique():,}",
)

with st.container(border=True):
    if alerts.empty:
        st.success("No negative reviews meet the selected confidence threshold.")
    else:
        st.dataframe(
            alerts_display[
                [
                    "date",
                    "brand",
                    "product",
                    "topic",
                    "confidence_pct",
                    "text",
                ]
            ],
            hide_index=True,
            use_container_width=True,
            column_config={
                "date": st.column_config.DatetimeColumn(
                    "Date",
                    format="DD MMM YYYY",
                ),
                "confidence_pct": st.column_config.NumberColumn(
                    "Confidence",
                    format="%.1f%%",
                ),
                "text": st.column_config.TextColumn(
                    "Review",
                    width="large",
                ),
            },
        )
