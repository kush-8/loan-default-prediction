"""Page 2: Data — Dataset profile, class imbalance, missing values."""

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    AMBER,
    GREEN,
    RED,
    base_fig_layout,
    load_test_sample,
    section_header,
)


def render():
    st.markdown("# 📊 Dataset Overview")
    st.caption(
        "Home Credit Default Risk dataset (Kaggle, 2018). Statistics from the committed test sample."
    )

    # ── Dataset dimensions (from metadata, hardcoded from training) ────────────
    section_header("Dataset Dimensions")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Training rows", "215,258", "70% of 307,511")
    with c2:
        st.metric("Validation rows", "46,127", "15%")
    with c3:
        st.metric("Test rows", "46,126", "15% — held out")
    with c4:
        st.metric("Raw features", "122", "+ 7 tables aggregated")

    st.markdown("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Historical tables", "6", "Bureau, POS, CC, Instal…")
    with c2:
        st.metric("Final features", "250", "After FE + historical join")
    with c3:
        st.metric("Positive rate", "8.07%", "Severely imbalanced")
    with c4:
        st.metric("Missing values", "~40-60%", "Many columns — LightGBM handles natively")

    # ── Class imbalance ────────────────────────────────────────────────────────
    section_header("Class Imbalance")
    sample = load_test_sample()

    col_l, col_r = st.columns([2, 3])
    with col_l:
        # Donut chart
        pos_rate = 0.0807
        fig = go.Figure(
            go.Pie(
                labels=["No default (0)", "Default (1)"],
                values=[1 - pos_rate, pos_rate],
                hole=0.55,
                marker_colors=[GREEN, RED],
                textinfo="label+percent",
                textfont=dict(size=13, color="#e2e8f0"),
                hovertemplate="%{label}: %{value:.1%}<extra></extra>",
            )
        )
        fig.update_layout(
            paper_bgcolor="#0f1117",
            plot_bgcolor="#0f1117",
            font=dict(color="#9ca3af"),
            height=280,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#e2e8f0")),
        )
        fig.add_annotation(
            text="8.1%<br>default",
            x=0.5,
            y=0.5,
            font=dict(size=18, color="#e2e8f0"),
            showarrow=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("""
**Why imbalance matters:**

A naïve model that predicts "no default" for everyone achieves **91.9% accuracy**
while being completely useless for risk management.

This is why the project uses:
- **Stratified splits** — preserves 8.1% positive rate in train/val/test
- **PR-AUC** instead of ROC-AUC as the primary metric (less optimistic on imbalanced data)
- **F1-optimal threshold** (~0.1479) instead of the default 0.5
- LightGBM with `is_unbalance=True` / cost-weighted objective

**Implication for threshold choice:**

At the default 0.5 threshold, the model flags almost **zero** applicants as defaulters
because base probabilities cluster below 0.1479. The selected threshold was tuned
on the validation set to maximise F1.
        """)

    # ── Missing values heatmap ─────────────────────────────────────────────────
    section_header("Feature Missingness (Test Sample)")
    if sample is not None:
        X = sample.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
        miss_pct = (X.isnull().mean() * 100).sort_values(ascending=False)
        top_miss = miss_pct[miss_pct > 0].head(30)

        if not top_miss.empty:
            fig = px.bar(
                x=top_miss.values,
                y=top_miss.index,
                orientation="h",
                color=top_miss.values,
                color_continuous_scale=[[0, GREEN], [0.3, AMBER], [1, RED]],
                labels={"x": "Missing %", "y": "Feature"},
                title="Top 30 Features by Missingness Rate",
            )
            base_fig_layout(fig, height=500)
            fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"{len(miss_pct[miss_pct > 0])} of {len(miss_pct)} features have missing values. "
                "LightGBM handles missing values natively without imputation."
            )
        else:
            st.info("No missing values in the test sample.")
    else:
        st.warning("Test sample not found at `data/processed/test_sample.csv`.")

    # ── Feature types ─────────────────────────────────────────────────────────
    if sample is not None:
        section_header("Feature Types")
        X = sample.drop(columns=["TARGET", "SK_ID_CURR"], errors="ignore")
        dtypes = X.dtypes.map(lambda d: "Categorical" if d == "object" else "Numeric")
        counts = dtypes.value_counts()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Numeric features", int(counts.get("Numeric", 0)))
        with c2:
            st.metric("Categorical features", int(counts.get("Categorical", 0)))
