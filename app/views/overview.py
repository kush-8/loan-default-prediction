"""Page 1: Project Overview — KPI cards, problem description, architecture."""

import streamlit as st

from app.utils import (
    AMBER,
    GREEN,
    PURPLE,
    kpi_card,
    load_metadata,
    section_header,
)


def render():
    meta = load_metadata()
    test = meta.get("test_metrics", {})
    thresh = meta.get("threshold", {})

    st.markdown("# 🏦 Loan Default Risk Prediction")
    st.markdown(
        "A production-grade, end-to-end ML system predicting the probability a loan applicant "
        "will default — covering leakage-free feature engineering, robust evaluation, "
        "calibration, SHAP explainability, a FastAPI inference service, and CI/CD automation."
    )

    # ── KPI Cards ──────────────────────────────────────────────────────────────
    section_header("Model Performance (Test Set)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(
            kpi_card("ROC-AUC", f"{test.get('roc_auc', 0):.4f}", "↑ vs random 0.50", PURPLE),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            kpi_card("PR-AUC", f"{test.get('pr_auc', 0):.4f}", "Baseline: 0.081", GREEN),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            kpi_card("Brier Score", f"{test.get('brier_score', 0):.4f}", "Lower is better", AMBER),
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            kpi_card(
                "F1 Score",
                f"{test.get('f1', 0):.4f}",
                f"@ threshold {thresh.get('selected', 0):.4f}",
                GREEN,
            ),
            unsafe_allow_html=True,
        )
    with c5:
        st.markdown(
            kpi_card("Recall", f"{test.get('recall', 0):.4f}", "Minority class", AMBER),
            unsafe_allow_html=True,
        )

    st.markdown("")

    # ── Problem Statement ──────────────────────────────────────────────────────
    section_header("Why This Problem Matters")
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("""
Banks extending credit to people without conventional history face a critical tradeoff:

| Error Type | Description | Cost |
|---|---|---|
| **False Negative** | Lend to someone who defaults | 💸 Full credit loss |
| **False Positive** | Reject a creditworthy borrower | 📉 Lost revenue + reputational risk |

These costs are **not equal**. In lending, a missed default typically costs **5–10× more**
than a wrongly rejected applicant. This is why:
- **Threshold selection** is a business lever, not a technical constant
- **PR-AUC** matters more than accuracy on an imbalanced dataset (8.1% positive rate)
- **Calibrated probabilities** enable risk-adjusted pricing, not just binary accept/reject
        """)
    with col_r:
        st.markdown("""
**Dataset**: Home Credit Default Risk (Kaggle, 2018)

| Table | Rows |
|---|---|
| Applications | 307,511 |
| Bureau records | 1.7M |
| Bureau balance | 27M |
| Prior applications | 1.7M |
| Instalments | 13.6M |
| POS/Cash balance | 10M |
| Credit card balance | 3.8M |

**Target**: `TARGET = 1` → payment difficulties  
**Class imbalance**: 8.1% positive rate
        """)

    # ── Architecture ──────────────────────────────────────────────────────────
    section_header("System Architecture")
    st.markdown("""
```
Raw Data ──► Offline Historical Aggregation ──► historical_features.parquet
     │                                                       │
     └──────────► FullFeatureEngineering.fit_transform() ◄───┘
                              │
                    ColumnTransformer (preprocessor)
                              │
                    LGBMClassifier (766 estimators, Optuna-tuned)
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        Threshold Selection          Calibration Check
        (val set only)               (val set only)
                 │                         │
                 └────────────┬────────────┘
                              ▼
                   Final Evaluation (test set, once)
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
        models/final_pipeline.joblib   model_metadata.json
                 │
                 ▼
        FastAPI  /v1/predict  ──►  Docker Container
```
    """)

    # ── Tech Stack ──────────────────────────────────────────────────────────
    section_header("Tech Stack")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**ML**")
        st.markdown("LightGBM · scikit-learn · SHAP · Optuna")
    with c2:
        st.markdown("**API**")
        st.markdown("FastAPI · Pydantic v2 · Uvicorn")
    with c3:
        st.markdown("**Infra**")
        st.markdown("Docker · GitHub Actions · pytest")
    with c4:
        st.markdown("**Dashboard**")
        st.markdown("Streamlit · Plotly")

    # ── Model metadata strip ────────────────────────────────────────────────
    with st.expander("📋 Full Model Metadata", expanded=False):
        st.json(meta)
