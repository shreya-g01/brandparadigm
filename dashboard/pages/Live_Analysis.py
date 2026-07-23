"""Live sentiment analysis using the deployed BrandParadigm API."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components.api_client import (
    API_BASE_URL,
    DashboardAPIError,
    analyze_dataframe,
    analyze_review,
    get_model_health,
)
from dashboard.components.page_shell import configure_page


configure_page("Live Analysis")
st.caption("Analyze customer reviews with the connected sentiment model.")

status_col, service_col = st.columns([1, 2])

with status_col:
    check_connection = st.button(
        "Check model connection",
        use_container_width=True,
    )

with service_col:
    st.caption(f"Connected service: {API_BASE_URL}")

if check_connection:
    try:
        with st.spinner("Checking the deployed model..."):
            health = get_model_health()

        models = health.get("models", [])
        model = models[0] if models else {}

        if model.get("loaded", False):
            st.success(
                "Sentiment model online"
                f" · {model.get('version', 'sentiment-baseline')}"
            )
        else:
            st.warning(
                model.get("error")
                or "The service responded, but the model is unavailable."
            )

    except DashboardAPIError as exc:
        st.warning(f"Could not reach the sentiment service: {exc}")


with st.container(border=True):
    st.subheader("Analyze one review")

    with st.form("review-form"):
        review_text = st.text_area(
            "Customer review",
            placeholder=(
                "The battery lasts all day, but delivery was slow."
            ),
            height=130,
        )

        brand_col, product_col, source_col = st.columns(3)

        with brand_col:
            brand = st.text_input(
                "Brand",
                value="Unknown",
            )

        with product_col:
            product = st.text_input(
                "Product",
                value="Unknown",
            )

        with source_col:
            source = st.text_input(
                "Source",
                value="Manual",
            )

        submitted = st.form_submit_button(
            "Analyze review",
            use_container_width=True,
        )

    if submitted:
        if not review_text.strip():
            st.warning("Enter a customer review before analyzing.")

        else:
            try:
                with st.spinner("Analyzing the review..."):
                    prediction = analyze_review(
                        review_text,
                        brand,
                        product,
                        source,
                    )

                sentiment_col, confidence_col = st.columns(2)

                sentiment_col.metric(
                    "Sentiment",
                    prediction["sentiment"],
                )

                confidence_col.metric(
                    "Confidence",
                    f"{prediction['sentiment_confidence']:.1%}",
                )

                scores = prediction.get("sentiment_scores", {})
                if scores:
                    score_frame = pd.DataFrame(
                        {
                            "Sentiment": list(scores),
                            "Probability": list(scores.values()),
                        }
                    )
                    st.dataframe(
                        score_frame,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Probability": st.column_config.ProgressColumn(
                                "Probability",
                                min_value=0.0,
                                max_value=1.0,
                                format="percent",
                            ),
                        },
                    )

            except DashboardAPIError as exc:
                st.warning(f"Review analysis failed: {exc}")


with st.container(border=True):
    st.subheader("Analyze a CSV")
    st.caption(
        "Upload a CSV containing a text column. "
        "Brand, product, source, and date are optional."
    )

    uploaded_file = st.file_uploader(
        "Customer-review CSV",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            uploaded_dataframe = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.warning(f"Could not read the uploaded CSV: {exc}")
        else:
            st.dataframe(
                uploaded_dataframe.head(20),
                hide_index=True,
                use_container_width=True,
            )

            analyze_upload = st.button(
                "Analyze uploaded reviews",
                use_container_width=True,
            )

            if analyze_upload:
                try:
                    with st.spinner(
                        f"Analyzing {len(uploaded_dataframe):,} reviews..."
                    ):
                        results = analyze_dataframe(uploaded_dataframe)

                    st.session_state["live_results"] = results
                    st.success(
                        f"Analyzed {len(results):,} reviews."
                    )

                except (DashboardAPIError, ValueError) as exc:
                    st.warning(f"CSV analysis failed: {exc}")


if "live_results" in st.session_state:
    results = st.session_state["live_results"]

    with st.container(border=True):
        st.subheader("Analysis results")
        st.dataframe(
            results,
            hide_index=True,
            use_container_width=True,
        )

        st.download_button(
            "Download analyzed results",
            results.to_csv(index=False),
            "brandparadigm_analysis.csv",
            "text/csv",
            use_container_width=True,
        )
