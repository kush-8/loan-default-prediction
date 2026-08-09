"""
app/utils.py
============
Shared data-loading utilities for the Streamlit dashboard.

All loaders use st.cache_resource / st.cache_data to ensure objects are
loaded once per session. Streamlit re-uses cached results across rerenders.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# ── Colors ────────────────────────────────────────────────────────────────────
PURPLE = "#6366f1"
PURPLE_LIGHT = "#a78bfa"
GREEN = "#10b981"
AMBER = "#f59e0b"
RED = "#ef4444"
BLUE = "#3b82f6"
GRAY = "#6b7280"

PLOTLY_TEMPLATE = "plotly_dark"
PLOTLY_PAPER_BG = "#0f1117"
PLOTLY_PLOT_BG = "#1a1d27"

PALETTE = [PURPLE, GREEN, AMBER, RED, BLUE, "#ec4899", "#14b8a6", "#f97316"]


# ── Loaders ───────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading model metadata…")
def load_metadata() -> dict:
    path = PROJECT_ROOT / "models" / "model_metadata.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_resource(show_spinner="Loading evaluation artifacts…")
def load_artifacts() -> dict:
    path = PROJECT_ROOT / "reports" / "evaluation_artifacts.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner="Loading test sample…")
def load_test_sample() -> pd.DataFrame | None:
    path = PROJECT_ROOT / "data" / "processed" / "test_sample.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner="Loading config…")
def load_config() -> dict:
    path = PROJECT_ROOT / "config" / "config.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@st.cache_resource(show_spinner="Loading pipeline…")
def load_pipeline():
    """Load the trained sklearn pipeline (optional — for live prediction)."""
    try:
        import joblib

        path = PROJECT_ROOT / "models" / "final_pipeline.joblib"
        if not path.exists():
            return None
        return joblib.load(path)
    except Exception as exc:
        logger.warning(f"Pipeline load failed: {exc}")
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────


def kpi_card(label: str, value: str, delta: str = "", color: str = PURPLE) -> str:
    """Return an HTML KPI card string."""
    delta_html = (
        f"<div style='font-size:0.75rem;color:#9ca3af;margin-top:0.2rem'>{delta}</div>"
        if delta
        else ""
    )
    return f"""
    <div class='kpi-card'>
        <div class='kpi-value' style='background:linear-gradient(135deg,{color},{PURPLE_LIGHT});
            -webkit-background-clip:text;-webkit-text-fill-color:transparent'>{value}</div>
        <div class='kpi-label'>{label}</div>
        {delta_html}
    </div>
    """


def demo_banner() -> None:
    pass


def portfolio_banner() -> None:
    pass


def section_header(title: str) -> None:
    st.markdown(f"<div class='section-header'>{title}</div>", unsafe_allow_html=True)


def base_fig_layout(fig, title: str = "", height: int = 400):
    """Apply consistent dark layout to a Plotly figure."""

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        paper_bgcolor=PLOTLY_PAPER_BG,
        plot_bgcolor=PLOTLY_PLOT_BG,
        font=dict(color="#9ca3af", size=12),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(26,29,39,0.8)",
            bordercolor="#3d4166",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(gridcolor="#2d3147", zerolinecolor="#3d4166")
    fig.update_yaxes(gridcolor="#2d3147", zerolinecolor="#3d4166")
    return fig
