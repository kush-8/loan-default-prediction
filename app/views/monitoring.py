"""Page 9: Monitoring — Simulated drift dashboard (clearly labelled as demo)."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils import (
    AMBER,
    GREEN,
    PURPLE,
    RED,
    base_fig_layout,
    section_header,
)
from src.monitoring.demo_data import (
    generate_feature_drift_report,
    generate_performance_timeseries,
    generate_prediction_volume_timeseries,
    generate_score_drift_timeseries,
    get_demo_monitoring_summary,
)


def render():
    st.markdown("# 📡 Model Monitoring")
    st.caption("Demonstration of a production monitoring system using simulated data.")

    summary = get_demo_monitoring_summary()
    st.info(f"📖 **Scenario**: {summary['scenario']}")

    # ── Controls ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("**Monitoring Controls**")
        drift_week = st.slider("Drift event week", min_value=8, max_value=18, value=14)
        n_weeks = st.slider("Weeks to display", min_value=10, max_value=26, value=20)

    score_ts = generate_score_drift_timeseries(n_weeks=n_weeks, drift_week=drift_week)
    perf_ts = generate_performance_timeseries(n_weeks=n_weeks, drift_week=drift_week)
    vol_ts = generate_prediction_volume_timeseries(n_weeks=n_weeks)
    feat_drift = generate_feature_drift_report()

    score_df = pd.DataFrame(score_ts)
    perf_df = pd.DataFrame(perf_ts)
    vol_df = pd.DataFrame(vol_ts)

    # ── Status strip ────────────────────────────────────────────────────────────
    section_header("Current Status")
    c1, c2, c3, c4 = st.columns(4)
    last_psi = score_df["psi"].iloc[-1]
    last_ks = score_df["ks_statistic"].iloc[-1]
    last_roc = perf_df["roc_auc"].iloc[-1]
    last_vol = vol_df["volume"].iloc[-1]

    psi_delta = (
        "🟢 Stable" if last_psi < 0.10 else ("🟡 Moderate" if last_psi < 0.25 else "🔴 Drift!")
    )
    with c1:
        st.metric("Score PSI (latest week)", f"{last_psi:.4f}", psi_delta)
    with c2:
        st.metric("KS Statistic", f"{last_ks:.4f}", "↓ lower is better")
    with c3:
        st.metric("ROC-AUC (latest)", f"{last_roc:.4f}")
    with c4:
        st.metric("Weekly predictions", f"{last_vol:,}")

    # ── PSI drift timeseries ────────────────────────────────────────────────────
    section_header("Prediction Score Drift Over Time")
    fig = go.Figure()

    # PSI line
    fig.add_trace(
        go.Scatter(
            x=score_df["week_label"],
            y=score_df["psi"],
            mode="lines+markers",
            name="PSI",
            line=dict(color=PURPLE, width=2.5),
            marker=dict(size=6),
        )
    )
    # Threshold lines
    fig.add_hline(
        y=0.10,
        line_dash="dot",
        line_color=AMBER,
        annotation_text="Moderate (0.10)",
        annotation_font_color=AMBER,
    )
    fig.add_hline(
        y=0.25,
        line_dash="dot",
        line_color=RED,
        annotation_text="Significant (0.25)",
        annotation_font_color=RED,
    )

    # Shade the drift period
    fig.add_vrect(
        x0=f"Week {drift_week}",
        x1=f"Week {min(drift_week + 3, n_weeks)}",
        fillcolor="rgba(239,68,68,0.1)",
        layer="below",
        line_width=0,
        annotation_text="Drift event",
        annotation_position="top left",
        annotation_font_color=RED,
    )

    base_fig_layout(fig, "Score PSI — Weekly", height=360)
    fig.update_xaxes(title="Week")
    fig.update_yaxes(title="PSI Value")
    st.plotly_chart(fig, use_container_width=True)

    # ── Performance degradation ─────────────────────────────────────────────────
    section_header("Model Performance Over Time")
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=perf_df["week_label"],
            y=perf_df["roc_auc"],
            mode="lines+markers",
            name="ROC-AUC",
            line=dict(color=PURPLE, width=2),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=perf_df["week_label"],
            y=perf_df["f1"],
            mode="lines+markers",
            name="F1",
            line=dict(color=GREEN, width=2),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=perf_df["week_label"],
            y=perf_df["brier_score"],
            mode="lines+markers",
            name="Brier",
            line=dict(color=AMBER, width=2, dash="dash"),
        )
    )
    fig2.add_vrect(
        x0=f"Week {drift_week}",
        x1=f"Week {min(drift_week + 3, n_weeks)}",
        fillcolor="rgba(239,68,68,0.1)",
        layer="below",
        line_width=0,
    )
    base_fig_layout(fig2, "Model Metrics — Weekly", height=350)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Feature drift heatmap ───────────────────────────────────────────────────
    section_header("Per-Feature Drift Report")
    feat_names = list(feat_drift.keys())
    psi_vals = [feat_drift[f]["psi"] for f in feat_names]
    ks_vals = [feat_drift[f]["ks_statistic"] for f in feat_names]

    feat_df = pd.DataFrame(
        {
            "Feature": feat_names,
            "PSI": psi_vals,
            "KS Statistic": ks_vals,
            "Status": [feat_drift[f]["psi_status"].replace("_", " ").title() for f in feat_names],
        }
    ).sort_values("PSI", ascending=False)

    # Color-code rows
    def _color_psi(val):
        if val < 0.10:
            return "color: #6ee7b7"
        elif val < 0.25:
            return "color: #fde68a"
        return "color: #fca5a5; font-weight: bold"

    st.dataframe(
        feat_df.style.applymap(_color_psi, subset=["PSI"]),
        use_container_width=True,
        hide_index=True,
        height=350,
    )

    # PSI bar chart
    fig3 = px.bar(
        feat_df,
        x="PSI",
        y="Feature",
        orientation="h",
        color="PSI",
        color_continuous_scale=[[0, GREEN], [0.4, AMBER], [1, RED]],
        title="Feature PSI — Current Week vs Reference",
    )
    base_fig_layout(fig3, height=400)
    fig3.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    fig3.add_vline(x=0.10, line_dash="dot", line_color=AMBER)
    fig3.add_vline(x=0.25, line_dash="dot", line_color=RED)
    st.plotly_chart(fig3, use_container_width=True)

    # ── Alert summary ─────────────────────────────────────────────────────────
    section_header("Alert Summary")
    significant = [f for f in feat_names if feat_drift[f]["psi_status"] == "significant_shift"]
    moderate = [f for f in feat_names if feat_drift[f]["psi_status"] == "moderate_shift"]

    if significant:
        st.error(f"🔴 **Significant drift** (PSI > 0.25): {', '.join(significant)}")
    if moderate:
        st.warning(f"🟡 **Moderate drift** (PSI 0.10–0.25): {', '.join(moderate)}")
    if not significant and not moderate:
        st.success("✅ All features stable (PSI < 0.10)")

    st.markdown(f"""
**Recommended action**: {summary['recommended_action']}
    """)

    st.caption(
        "⚠️ All values on this page are generated by `src/monitoring/demo_data.py` "
        "and do not represent real production traffic."
    )
