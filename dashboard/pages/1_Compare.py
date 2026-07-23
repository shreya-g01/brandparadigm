"""Brand comparison page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from dashboard.components.page_shell import (
    CHART_CONFIG,
    SKY,
    configure_page,
    load_reviews,
    transparent_chart_layout,
)


configure_page("Compare")
st.caption("Compare review volume, positive sentiment, and confidence by brand.")

reviews = load_reviews()
brands = sorted(reviews["brand"].dropna().astype(str).unique())
selected_brands = st.multiselect(
    "Brands",
    brands,
    default=brands,
)

filtered = reviews[reviews["brand"].isin(selected_brands)]
summary = (
    filtered
    .assign(positive=filtered["sentiment"].eq("Positive"))
    .groupby("brand", as_index=False)
    .agg(
        reviews=("text", "size"),
        positive_rate=("positive", "mean"),
        average_confidence=("confidence", "mean"),
    )
)
summary["positive_pct"] = summary["positive_rate"] * 100
summary["confidence_pct"] = summary["average_confidence"] * 100
summary = summary.sort_values("positive_pct", ascending=True)

if summary.empty:
    st.info("Select at least one brand.")
else:
    with st.container(border=True):
        figure = px.bar(
            summary,
            x="positive_pct",
            y="brand",
            orientation="h",
            text="positive_pct",
            hover_data={
                "reviews": ":,",
                "average_confidence": ":.1%",
                "positive_pct": ":.1f",
            },
        )
        figure.update_traces(
            marker_color=SKY,
            texttemplate="%{x:.1f}%",
            textposition="outside",
            cliponaxis=False,
        )
        figure.update_xaxes(
            title="Positive sentiment",
            range=[0, 105],
            ticksuffix="%",
        )
        figure.update_yaxes(title=None)
        transparent_chart_layout(figure, height=430)
        st.plotly_chart(
            figure,
            use_container_width=True,
            config=CHART_CONFIG,
        )

    with st.container(border=True):
        st.subheader("Brand comparison details")
        display = summary[
            [
                "brand",
                "reviews",
                "positive_pct",
                "confidence_pct",
            ]
        ].sort_values("positive_pct", ascending=False)
        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "brand": "Brand",
                "reviews": st.column_config.NumberColumn(
                    "Reviews",
                    format="%d",
                ),
                "positive_pct": st.column_config.NumberColumn(
                    "Positive",
                    format="%.1f%%",
                ),
                "confidence_pct": st.column_config.NumberColumn(
                    "Avg. confidence",
                    format="%.1f%%",
                ),
            },
        )
