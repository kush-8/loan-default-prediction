"""
app/streamlit_app.py
====================
Loan Default Risk Prediction — Interactive Dashboard

Run from the project root:
    streamlit run app/streamlit_app.py

Pages
-----
1.  Project Overview   — KPI cards, problem description, architecture
2.  Data               — Dataset profile, class imbalance, missing values
3.  Feature Engineering — Pipeline flow, engineered features, leakage prevention
4.  Model Comparison   — LR vs RF vs LightGBM benchmark
5.  Performance        — ROC/PR curves, interactive threshold slider
6.  Calibration        — Reliability diagram, Brier scores
7.  Explainability     — SHAP global and local explanations
8.  MLOps Architecture — Pipeline diagram, versioning, CI/CD
9.  Monitoring         — Demo drift dashboard (labelled as demonstration data)
10. Live Prediction    — Form-based applicant scoring
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.*` and `app.*` imports work
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st  # noqa: E402

# ── Page configuration (MUST be first Streamlit call) ─────────────────────────
st.set_page_config(
    page_title="Loan Default Risk Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/kush-8/loan-default-prediction",
        "Report a bug": "https://github.com/kush-8/loan-default-prediction/issues",
        "About": "A portfolio ML engineering project — Home Credit Default Risk.",
    },
)

# ── Global styles ──────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .sidebar-title {
        font-size: 1.3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2d3147;
        margin-bottom: 1rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
        border: 1px solid #3d4166;
        border-radius: 12px;
        padding: 1.25rem 1rem;
        text-align: center;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #6366f1;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.78rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }
    .section-header {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
        border-left: 4px solid #6366f1;
        padding-left: 0.75rem;
        margin: 1.5rem 0 1rem 0;
    }
    .demo-banner {
        background: linear-gradient(135deg, #451a03 0%, #78350f 100%);
        border: 1px solid #b45309;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: #fde68a;
        margin-bottom: 1rem;
    }
    .info-banner {
        background: linear-gradient(135deg, #1e1b4b 0%, #2e2d5e 100%);
        border: 1px solid #4338ca;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-size: 0.85rem;
        color: #c7d2fe;
        margin-bottom: 1rem;
    }
    .risk-low    { background: #064e3b; color: #6ee7b7; }
    .risk-medium { background: #1e3a5f; color: #93c5fd; }
    .risk-high   { background: #78350f; color: #fde68a; }
    .risk-vhigh  { background: #7f1d1d; color: #fca5a5; }
    .risk-badge {
        display: inline-block;
        border-radius: 20px;
        padding: 0.35rem 1rem;
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 0.05em;
    }
    [data-testid="stTabs"] button { font-weight: 600; color: #9ca3af; }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #a78bfa;
        border-bottom-color: #6366f1;
    }
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Navigation ─────────────────────────────────────────────────────────────────
from app.views import (  # noqa: E402
    calibration,
    data,
    explainability,
    feature_engineering,
    live_prediction,
    mlops,
    model_comparison,
    monitoring,
    overview,
    performance,
)

PAGES = {
    "📋 Project Overview": overview,
    "📊 Data": data,
    "⚙️ Feature Engineering": feature_engineering,
    "🏆 Model Comparison": model_comparison,
    "📈 Performance": performance,
    "🎯 Calibration": calibration,
    "🔍 Explainability": explainability,
    "🔧 MLOps Architecture": mlops,
    "📡 Monitoring": monitoring,
    "🔮 Live Prediction": live_prediction,
}

with st.sidebar:
    st.markdown(
        "<div class='sidebar-title'>🏦 Loan Default Risk</div>",
        unsafe_allow_html=True,
    )
    st.caption("Home Credit Default Risk")
    st.markdown("---")

    selection = st.radio(
        "Navigate to",
        list(PAGES.keys()),
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption("**Model Version:** 1.1.0")
    st.caption("**Dataset:** Home Credit (2018)")
    st.caption("**ROC-AUC (test):** 0.7720")

# ── Render selected page ───────────────────────────────────────────────────────
PAGES[selection].render()
