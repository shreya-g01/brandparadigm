"""Shared presentation helpers for BrandParadigm dashboard pages."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.data import load_dashboard_data  # noqa: E402


SKY = "#68B6D9"
BRONZE = "#A66A32"
TEAL = "#3A8DA8"
INK = "#17324D"
MUTED = "#6D8493"
GRID = "#DBE8EE"

CHART_CONFIG = {
    "displayModeBar": False,
    "displaylogo": False,
    "responsive": True,
}

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700&display=swap');
:root {
    --ink:#17324d;
    --muted:#6d8493;
    --line:#dbe8ee;
    --sky:#68b6d9;
    --bronze:#a66a32;
    --teal:#3a8da8;
}
.stApp {
    background:linear-gradient(145deg,#f7fbfd 0%,#f3f7f9 52%,#fbf8f4 100%);
    color:var(--ink);
    font-family:'DM Sans',sans-serif;
}
[data-testid="stHeader"] {
    background:rgba(247,251,253,.9);
}
[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#eaf6fb 0%,#f5fafc 55%,#f5eee7 100%);
    border-right:1px solid #d5e6ed;
}
[data-testid="stSidebar"] * {
    color:#17324d !important;
}
[data-testid="stSidebarNavLink"][aria-current="page"] {
    background:#d9eef7 !important;
    border:1px solid #68b6d9 !important;
    border-radius:10px !important;
}
[data-testid="stSidebarNavLink"][aria-current="page"] span {
    color:#17324d !important;
    font-weight:700 !important;
}
.block-container {
    max-width:1480px;
    padding-top:2rem;
    padding-bottom:3rem;
}
h1,h2,h3 {
    font-family:'Manrope',sans-serif !important;
    letter-spacing:-.035em;
}
[data-testid="stMetric"] {
    background:rgba(255,255,255,.9);
    border:1px solid var(--line);
    border-top:3px solid var(--sky);
    border-radius:16px;
    padding:1.05rem 1.15rem;
    box-shadow:0 8px 24px rgba(23,50,77,.055);
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background:rgba(255,255,255,.92);
    border-color:var(--line) !important;
    border-radius:18px;
    box-shadow:0 8px 24px rgba(23,50,77,.055);
}
.stMultiSelect [data-baseweb="tag"] {
    background-color:#d9eef7 !important;
    border:1px solid #68b6d9 !important;
    color:#17324d !important;
}
.stMultiSelect [data-baseweb="tag"] span,
.stMultiSelect [data-baseweb="tag"] svg {
    color:#17324d !important;
    fill:#17324d !important;
}
#MainMenu,footer {
    visibility:hidden;
}
</style>
"""


def configure_page(title: str) -> None:
    """Set page metadata and apply the shared dashboard theme."""

    st.set_page_config(
        page_title=f"BrandParadigm | {title}",
        page_icon="◈",
        layout="wide",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)
    st.title(title)


@st.cache_data(ttl=60)
def load_reviews() -> pd.DataFrame:
    """Return normalized review data for secondary dashboard pages."""

    reviews = load_dashboard_data().reviews.copy()
    reviews["date"] = pd.to_datetime(reviews["date"], errors="coerce")
    reviews["confidence"] = pd.to_numeric(
        reviews["confidence"],
        errors="coerce",
    ).fillna(0.0)
    return reviews


def transparent_chart_layout(figure, *, height: int = 480):
    """Apply the shared visual treatment to a Plotly figure."""

    figure.update_layout(
        height=height,
        margin=dict(l=8, r=28, t=18, b=42),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK),
        legend_title_text="",
    )
    figure.update_xaxes(
        gridcolor=GRID,
        zeroline=False,
    )
    figure.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
    )
    return figure
