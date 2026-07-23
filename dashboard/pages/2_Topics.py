"""Topic intelligence page."""

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


configure_page("Topics")
st.caption("See what customers discuss and where negative sentiment concentrates.")

reviews = load_reviews()
topic_summary = (
    reviews
    .assign(negative=reviews["sentiment"].eq("Negative"))
    .groupby("topic", as_index=False)
    .agg(
        mentions=("text", "size"),
        negative_rate=("negative", "mean"),
    )
    .sort_values("mentions", ascending=False)
)

top_n = st.slider(
    "Topics to display",
    min_value=5,
    max_value=min(25, max(5, len(topic_summary))),
    value=min(12, max(5, len(topic_summary))),
)

top_topics = (
    topic_summary
    .head(top_n)
    .sort_values("mentions", ascending=True)
)

chart_col, risk_col = st.columns([1.35, 1], gap="large")

with chart_col:
    with st.container(border=True):
        figure = px.bar(
            top_topics,
            x="mentions",
            y="topic",
            orientation="h",
            text="mentions",
        )
        figure.update_traces(
            marker_color=SKY,
            textposition="outside",
            cliponaxis=False,
        )
        figure.update_xaxes(title="Mentions")
        figure.update_yaxes(title=None)
        transparent_chart_layout(figure, height=540)
        st.plotly_chart(
            figure,
            use_container_width=True,
            config=CHART_CONFIG,
        )

with risk_col:
    with st.container(border=True):
        st.subheader("Negative-topic risk")
        risk = (
            topic_summary
            .sort_values(
                ["negative_rate", "mentions"],
                ascending=False,
            )
            .head(10)
            .copy()
        )
        risk["negative_pct"] = risk["negative_rate"] * 100
        st.dataframe(
            risk[["topic", "mentions", "negative_pct"]],
            hide_index=True,
            use_container_width=True,
            column_config={
                "topic": "Topic",
                "mentions": "Mentions",
                "negative_pct": st.column_config.ProgressColumn(
                    "Negative share",
                    min_value=0,
                    max_value=100,
                    format="%d%%",
                ),
            },
        )
        st.caption(
            "Higher negative share indicates a stronger customer-friction signal."
        )

with st.container(border=True):
    st.subheader("Topic review examples")
    selected_topic = st.selectbox(
        "Topic",
        topic_summary["topic"].tolist(),
    )
    examples = (
        reviews[reviews["topic"].eq(selected_topic)]
        .sort_values(["confidence", "date"], ascending=False)
        [["date", "brand", "sentiment", "confidence", "text"]]
        .head(20)
        .copy()
    )
    examples["confidence_pct"] = examples.pop("confidence") * 100
    st.dataframe(
        examples,
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
