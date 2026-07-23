"""Sentiment trend page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.components.page_shell import (
    BRONZE,
    CHART_CONFIG,
    SKY,
    configure_page,
    load_reviews,
    transparent_chart_layout,
)


configure_page("Trends")
st.caption("Track positive and negative conversation volume over time.")

reviews = load_reviews().dropna(subset=["date"])
brands = sorted(reviews["brand"].dropna().astype(str).unique())
selected_brands = st.multiselect(
    "Brands",
    brands,
    default=brands,
)

filtered = reviews[reviews["brand"].isin(selected_brands)].copy()
filtered["day"] = filtered["date"].dt.floor("D")

trend = (
    filtered
    .groupby(["day", "sentiment"], as_index=False)
    .size()
    .rename(columns={"size": "reviews"})
)

with st.container(border=True):
    if trend.empty:
        st.info("Select at least one brand.")
    else:
        figure = px.line(
            trend,
            x="day",
            y="reviews",
            color="sentiment",
            markers=True,
            color_discrete_map={
                "Positive": SKY,
                "Negative": BRONZE,
            },
        )
        figure.update_xaxes(title=None)
        figure.update_yaxes(
            title="Reviews",
            rangemode="tozero",
        )
        transparent_chart_layout(figure, height=500)
        st.plotly_chart(
            figure,
            use_container_width=True,
            config=CHART_CONFIG,
        )

with st.container(border=True):
    st.subheader("Daily sentiment data")
    pivot = (
        trend
        .pivot(
            index="day",
            columns="sentiment",
            values="reviews",
        )
        .fillna(0)
        .astype(int)
        .reset_index()
        .sort_values("day", ascending=False)
    )
    st.dataframe(
        pivot,
        hide_index=True,
        use_container_width=True,
        column_config={
            "day": st.column_config.DateColumn(
                "Date",
                format="DD MMM YYYY",
            ),
        },
    )
