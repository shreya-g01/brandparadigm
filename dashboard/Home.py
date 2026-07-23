"""BrandParadigm executive sentiment dashboard."""

from __future__ import annotations

from html import escape
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brandparadigm.config.settings import get_settings  # noqa: E402
from dashboard.components.charts import sentiment_donut, sentiment_trend, topic_breakdown  # noqa: E402
from dashboard.components.data import load_dashboard_data, load_recent_voice  # noqa: E402
from dashboard.components.api_client import (  # noqa: E402
    DashboardAPIError,
    analyze_dataframe,
    analyze_review,
    get_model_health,
)

st.set_page_config(page_title="BrandParadigm | Intelligence", page_icon="◈", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700&display=swap');
:root { --ink:#17324d; --muted:#6d8493; --line:#dbe8ee; --sky:#68b6d9; --bronze:#a66a32; }
.stApp { background:linear-gradient(145deg,#f7fbfd 0%,#f3f7f9 52%,#fbf8f4 100%); color:var(--ink); font-family:'DM Sans',sans-serif; }
[data-testid="stHeader"] { background:rgba(247,251,253,.9); }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#eaf6fb 0%,#f5fafc 55%,#f5eee7 100%); border-right:1px solid #d5e6ed; }
[data-testid="stSidebar"] * { color:#17324d !important; }
[data-testid="stSidebar"] hr { border-color:#d5e6ed; }
.block-container { max-width:1480px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; letter-spacing:-.035em; }
.eyebrow { color:#a66a32; font-weight:700; font-size:.76rem; letter-spacing:.14em; text-transform:uppercase; }
.hero { display:flex; justify-content:space-between; align-items:end; gap:2rem; margin:.25rem 0 1.4rem; }
.hero h1 { margin:.2rem 0 .35rem; font-size:2.25rem; }
.hero p { color:var(--muted); margin:0; }
.fresh { border:1px solid #c5e4f1; background:#eaf6fb; color:#296d8c; padding:.5rem .75rem; border-radius:999px; font-size:.78rem; font-weight:600; white-space:nowrap; }
[data-testid="stMetric"] { background:rgba(255,255,255,.9); border:1px solid var(--line); border-top:3px solid #68b6d9; border-radius:16px; padding:1.05rem 1.15rem; box-shadow:0 8px 24px rgba(23,50,77,.055); }
[data-testid="stMetricLabel"] { color:var(--muted); }
[data-testid="stMetricValue"] { font-family:'Manrope',sans-serif; letter-spacing:-.04em; }
[data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.92); border-color:var(--line) !important; border-radius:18px; box-shadow:0 8px 24px rgba(23,50,77,.055); }
.section-title { font-family:'Manrope'; font-size:1rem; font-weight:700; margin:0; }
.section-sub { color:var(--muted); font-size:.78rem; margin:.15rem 0 .4rem; }
.review { border-bottom:1px solid var(--line); padding:.8rem 0; }
.review:last-child { border-bottom:0; }
.review-head { display:flex; justify-content:space-between; font-size:.76rem; color:var(--muted); }
.review p { font-size:.9rem; margin:.4rem 0; line-height:1.5; }
.pill { display:inline-block; border-radius:999px; padding:.16rem .45rem; font-size:.67rem; font-weight:700; }
.positive { color:#296d8c; background:#e7f5fb; } .negative { color:#875126; background:#f5e9dd; }
.insight { background:linear-gradient(135deg,#17324d,#275978); color:white; padding:1rem 1.1rem; border-radius:14px; margin:.5rem 0; border-left:3px solid #b87b43; }
.insight small { color:#c9e8f4; text-transform:uppercase; letter-spacing:.08em; }
.insight p { margin:.4rem 0 0; line-height:1.45; font-size:.88rem; }
#MainMenu,footer { visibility:hidden; }
@media(max-width:760px){ .hero{align-items:start;flex-direction:column}.hero h1{font-size:1.8rem}.block-container{padding-top:1rem} }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60)
def get_data():
    return load_dashboard_data()


@st.cache_data(ttl=60)
def get_recent_voice():
    return load_recent_voice()


data = get_data()
reviews = data.reviews
recent_voice = get_recent_voice()
settings = get_settings()
CHART_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": True,
    "responsive": True,
    "modeBarButtonsToAdd": ["drawline", "drawrect", "eraseshape"],
}

with st.sidebar:
    st.markdown("## ◈ BrandParadigm")
    st.caption("COMPETITOR INTELLIGENCE")
    st.divider()
    st.markdown("#### Filter analysis")
    selected_brands = st.multiselect("Brand", sorted(reviews["brand"].unique()), placeholder="All brands")
    selected_products = st.multiselect("Product", sorted(reviews["product"].unique()), placeholder="All products")
    selected_sources = st.multiselect("Source", sorted(reviews["source"].unique()), placeholder="All sources")
    st.divider()
    st.caption("DATA CONNECTION")
    if data.mode == "demo":
        st.warning("Demo dataset active")
        st.caption("Add an analyzed CSV to replace it automatically.")
    else:
        st.success("Analysis output connected")
        st.caption(data.source_path)
    st.caption(f"API · {settings.dashboard_api_base_url}")
    st.divider()
    st.caption("MODEL STATUS")
    sentiment_ready = Path(settings.model_sentiment_path).exists()
    topic_ready = Path(settings.model_topic_model_path).exists()
    st.caption(f"{'●' if sentiment_ready else '○'} Sentiment model · {'ready' if sentiment_ready else 'artifact pending'}")
    st.caption(f"{'●' if topic_ready else '○'} Topic model · {'ready' if topic_ready else 'phase pending'}")

filtered = reviews.copy()
if selected_brands:
    filtered = filtered[filtered["brand"].isin(selected_brands)]
if selected_products:
    filtered = filtered[filtered["product"].isin(selected_products)]
if selected_sources:
    filtered = filtered[filtered["source"].isin(selected_sources)]

status_label = "Analysis output connected" if data.mode != "demo" else "Demo analysis active"
st.markdown(
    f"""<div class="hero"><div><div class="eyebrow">Sentiment command center</div>
    <h1>Brand intelligence, at a glance.</h1><p>Monitor customer perception, surface friction, and spot momentum across your portfolio.</p>
    </div><span class="fresh">● {status_label}</span></div>""",
    unsafe_allow_html=True,
)

with st.expander("Live model analysis", expanded=False):
    if st.button("Check API and model status"):
        try:
            health = get_model_health()
            for model in health.get("models", []):
                icon = "🟢" if model["loaded"] else "⚪"
                st.caption(
                    f"{icon} {model['name'].title()}: "
                    f"{model.get('version') or model.get('error')}"
                )
        except DashboardAPIError as exc:
            st.info(f"Start the API to enable live analysis. {exc}")

    with st.form("live-review"):
        live_text = st.text_area("Customer review")
        live_brand, live_product, live_source = st.columns(3)
        brand = live_brand.text_input("Brand", "Unknown")
        product = live_product.text_input("Product", "Unknown")
        source = live_source.text_input("Source", "Manual")
        submitted = st.form_submit_button("Analyze review")
    if submitted:
        if not live_text.strip():
            st.error("Enter a non-empty review.")
        else:
            try:
                result = analyze_review(live_text, brand, product, source)
                c1, c2, c3 = st.columns(3)
                c1.metric("Sentiment", result["sentiment"], f"{result['sentiment_confidence']:.1%}")
                c2.metric("Topic", result.get("topic") or "Unavailable")
                c3.metric("Category", result.get("category") or "Unavailable")
            except DashboardAPIError as exc:
                st.error(str(exc))

    upload = st.file_uploader("Analyze a CSV containing a text column", type=["csv"])
    if upload is not None and st.button("Analyze uploaded reviews"):
        try:
            st.session_state["live_results"] = analyze_dataframe(pd.read_csv(upload))
        except (DashboardAPIError, ValueError) as exc:
            st.error(str(exc))
    if "live_results" in st.session_state:
        live_results = st.session_state["live_results"]
        st.dataframe(live_results, use_container_width=True)
        st.download_button(
            "Download analyzed CSV",
            live_results.to_csv(index=False),
            "brandparadigm_analysis.csv",
            "text/csv",
        )

total = len(filtered)
positive = int((filtered["sentiment"] == "Positive").sum())
negative = int((filtered["sentiment"] == "Negative").sum())
confidence = filtered["confidence"].mean() if total else 0
k1, k2, k3, k4 = st.columns(4)
k1.metric("Reviews analyzed", f"{total:,}", help="Reviews matching the current filters")
k2.metric("Positive sentiment", f"{positive / total if total else 0:.0%}", f"{positive - negative:+d} net reviews")
k3.metric("Negative signals", f"{negative:,}", help="Items classified as negative")
k4.metric("Avg. confidence", f"{confidence:.0%}", help="Mean classifier confidence")

left, right = st.columns([1.05, 1.95])
with left, st.container(border=True):
    st.markdown('<div class="section-title">Sentiment mix</div><div class="section-sub">Current filtered distribution</div>', unsafe_allow_html=True)
    st.plotly_chart(sentiment_donut(filtered), use_container_width=True, config=CHART_CONFIG)
with right, st.container(border=True):
    st.markdown('<div class="section-title">Conversation momentum</div><div class="section-sub">Review volume by sentiment over time</div>', unsafe_allow_html=True)
    st.plotly_chart(sentiment_trend(filtered), use_container_width=True, config=CHART_CONFIG)

topics_col, insights_col = st.columns([1.7, 1])
with topics_col, st.container(border=True):
    st.markdown('<div class="section-title">What customers talk about</div><div class="section-sub">Topic volume split by sentiment</div>', unsafe_allow_html=True)
    st.plotly_chart(topic_breakdown(filtered), use_container_width=True, config=CHART_CONFIG)
with insights_col, st.container(border=True):
    st.markdown('<div class="section-title">Signals to watch</div><div class="section-sub">Generated from the current view</div>', unsafe_allow_html=True)
    if total:
        stats = filtered.assign(negative=filtered["sentiment"].eq("Negative")).groupby("topic").agg(volume=("text", "size"), negative_rate=("negative", "mean"))
        risk = stats.sort_values(["negative_rate", "volume"], ascending=False).iloc[0]
        strengths = filtered[filtered["sentiment"] == "Positive"]["topic"].value_counts()
        strength = strengths.index[0] if not strengths.empty else "No positive topic yet"
        st.markdown(f'<div class="insight"><small>Priority risk</small><p><b>{risk.name}</b> has a {risk.negative_rate:.0%} negative share across {int(risk.volume)} mentions.</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="insight"><small>Brand strength</small><p><b>{strength}</b> generates the most positive conversation in this view.</p></div>', unsafe_allow_html=True)
    else:
        st.info("No reviews match the selected filters.")

with st.container(border=True):
    st.markdown('<div class="section-title">Recent customer voice</div><div class="section-sub">Latest high-confidence reviews and social posts</div>', unsafe_allow_html=True)
    recent_feed = recent_voice.copy() if recent_voice is not None else filtered.copy()
    if selected_brands:
        recent_feed = recent_feed[recent_feed["brand"].isin(selected_brands)]
    if selected_sources:
        recent_feed = recent_feed[recent_feed["source"].isin(selected_sources)]
    recent_feed = recent_feed.sort_values(["date", "confidence"], ascending=False).head(6)
    if recent_feed.empty:
        st.info("No recent voice entries match the selected filters.")
    for row in recent_feed.itertuples():
        sentiment_class = "positive" if row.sentiment == "Positive" else "negative"
        st.markdown(
            f'<div class="review"><div class="review-head"><span><b>{escape(str(row.brand))}</b> · {escape(str(row.product))} · {escape(str(row.source))}</span><span>{pd.Timestamp(row.date):%d %b}</span></div>'
            f'<p>{escape(str(row.text))}</p><span class="pill {sentiment_class}">{escape(str(row.sentiment))} · {row.confidence:.0%}</span> '
            f'<span class="pill" style="background:#eef2f6;color:#475569">{escape(str(row.topic))}</span></div>',
            unsafe_allow_html=True,
        )

st.caption("BrandParadigm · Binary RoBERTa sentiment · Topic discovery and classifier status shown when artifacts are available")
