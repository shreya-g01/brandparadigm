"""Plotly chart builders kept separate from Streamlit page layout."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

NAVY = "#17324D"
SKY = "#68B6D9"
BRONZE = "#A66A32"
MUTED = "#9DAFBA"
GRID = "rgba(104, 182, 217, .18)"


def _finish(fig: go.Figure, height: int = 300) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#17324D", font_color="white", bordercolor=SKY),
        dragmode="zoom",
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def sentiment_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["sentiment"].value_counts().rename_axis("sentiment").reset_index(name="reviews")
    fig = px.pie(
        counts,
        names="sentiment",
        values="reviews",
        hole=0.72,
        color="sentiment",
        color_discrete_map={"Positive": SKY, "Negative": BRONZE, "Unknown": MUTED},
    )
    fig.update_traces(textinfo="percent", textfont_size=13, marker=dict(line=dict(color="white", width=4)))
    fig.add_annotation(text=f"<b>{len(df):,}</b><br><span style='font-size:11px'>reviews</span>", showarrow=False)
    return _finish(fig)


def sentiment_trend(df: pd.DataFrame) -> go.Figure:
    daily = (
        df.assign(day=pd.to_datetime(df["date"]).dt.date)
        .groupby(["day", "sentiment"])
        .size()
        .rename("reviews")
        .reset_index()
    )
    fig = px.line(
        daily,
        x="day",
        y="reviews",
        color="sentiment",
        color_discrete_map={"Positive": SKY, "Negative": BRONZE, "Unknown": MUTED},
        markers=True,
    )
    fig.update_traces(line_width=2.5, marker_size=5)
    return _finish(fig, 315)


def topic_breakdown(df: pd.DataFrame) -> go.Figure:
    grouped = df.groupby(["topic", "sentiment"]).size().rename("reviews").reset_index()
    topics = df["topic"].value_counts().index.tolist()
    fig = px.bar(
        grouped,
        x="reviews",
        y="topic",
        color="sentiment",
        orientation="h",
        category_orders={"topic": topics[::-1]},
        color_discrete_map={"Positive": SKY, "Negative": BRONZE, "Unknown": MUTED},
    )
    fig.update_layout(barmode="stack", showlegend=False)
    return _finish(fig, 320)
