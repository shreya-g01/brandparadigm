"""Downloadable dashboard reports page."""

from __future__ import annotations

import streamlit as st

from dashboard.components.page_shell import (
    configure_page,
    load_reviews,
)


configure_page("Reports")
st.caption("Export normalized review data and aggregated brand performance.")

reviews = load_reviews()
brand_report = (
    reviews
    .assign(
        positive=reviews["sentiment"].eq("Positive"),
        negative=reviews["sentiment"].eq("Negative"),
    )
    .groupby("brand", as_index=False)
    .agg(
        reviews=("text", "size"),
        positive_reviews=("positive", "sum"),
        negative_reviews=("negative", "sum"),
        positive_rate=("positive", "mean"),
        average_confidence=("confidence", "mean"),
    )
    .sort_values("reviews", ascending=False)
)
brand_report["positive_pct"] = (
    brand_report.pop("positive_rate") * 100
)
brand_report["confidence_pct"] = (
    brand_report.pop("average_confidence") * 100
)

with st.container(border=True):
    st.subheader("Brand performance report")
    st.dataframe(
        brand_report,
        hide_index=True,
        use_container_width=True,
        column_config={
            "positive_pct": st.column_config.NumberColumn(
                "Positive rate",
                format="%.1f%%",
            ),
            "confidence_pct": st.column_config.NumberColumn(
                "Avg. confidence",
                format="%.1f%%",
            ),
        },
    )

download_col_1, download_col_2 = st.columns(2)

with download_col_1:
    st.download_button(
        "Download review-level CSV",
        reviews.to_csv(index=False).encode("utf-8"),
        file_name="brandparadigm_reviews.csv",
        mime="text/csv",
        use_container_width=True,
    )

with download_col_2:
    st.download_button(
        "Download brand report CSV",
        brand_report.to_csv(index=False).encode("utf-8"),
        file_name="brandparadigm_brand_report.csv",
        mime="text/csv",
        use_container_width=True,
    )

with st.container(border=True):
    st.subheader("Review-level data")
    review_display = reviews.sort_values(
        "date",
        ascending=False,
    ).copy()
    review_display["confidence_pct"] = (
        review_display.pop("confidence") * 100
    )
    st.dataframe(
        review_display,
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
