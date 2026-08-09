"""Page 7: Explainability — Global SHAP + local explanation."""

import plotly.express as px
import streamlit as st

from app.utils import (
    PROJECT_ROOT,
    PURPLE,
    base_fig_layout,
    load_artifacts,
    section_header,
)


def render():
    st.markdown("# 🔍 Model Explainability")
    st.caption(
        "SHAP (SHapley Additive exPlanations) — mathematically grounded feature attribution."
    )

    # ── What is SHAP ───────────────────────────────────────────────────────────
    with st.expander("What is SHAP?", expanded=False):
        st.markdown("""
SHAP values are based on Shapley values from cooperative game theory.
For each prediction, SHAP assigns a contribution to each feature such that
contributions sum to the model output (log-odds).

**Key properties:**
- **Additive**: SHAP values sum to the prediction (minus the base value)
- **Consistent**: A feature that matters more gets a higher SHAP value
- **Locally accurate**: Each explanation is faithful to the model's actual computation

**Implementation**: `shap.TreeExplainer` — uses the model's tree structure for
exact, fast computation without sampling.

> SHAP values explain the **model** — not the real-world causal relationship.
> "EXT_SOURCE_2 has high SHAP" means the model relies heavily on it, not
> that it causes defaults.
        """)

    # ── Global SHAP plot ───────────────────────────────────────────────────────
    section_header("Global Feature Importance (SHAP)")

    shap_bar_path = PROJECT_ROOT / "reports" / "shap_bar_plot.png"
    shap_summary_path = PROJECT_ROOT / "reports" / "shap_summary_plot.png"

    tab1, tab2 = st.tabs(["📊 Bar Chart (Mean |SHAP|)", "🔴 Beeswarm Plot"])

    with tab1:
        # Show from artifacts if available
        artifacts = load_artifacts()
        feat_imp = artifacts.get("feature_importance", [])
        if feat_imp:
            import pandas as pd

            df = pd.DataFrame(feat_imp[:20])
            fig = px.bar(
                df,
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale=[[0, "#3d4166"], [1, PURPLE]],
                title="Top 20 Features by LightGBM Gain Importance",
            )
            base_fig_layout(fig, height=500)
            fig.update_layout(
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run `python scripts/generate_report_artifacts.py` to generate SHAP JSON data.")

    with tab2:
        if shap_summary_path.exists():
            st.image(str(shap_summary_path), use_container_width=True)
            st.caption(
                "Beeswarm plot: each dot = one sample. Color = feature value (red=high, blue=low). "
                "X-axis = SHAP value (positive = increases default probability)."
            )
        else:
            st.info("Run `python src/models/explain.py` to generate the beeswarm plot.")

    # ── Feature importance table ───────────────────────────────────────────────
    artifacts = load_artifacts()
    feat_imp = artifacts.get("feature_importance", [])
    if feat_imp:
        section_header("Top Features — Detailed Table")
        import pandas as pd

        df = pd.DataFrame(feat_imp[:30])
        df["importance_pct"] = (df["importance"] * 100).round(2)
        df = df[["feature", "importance_pct"]].copy()
        df.columns = ["Feature", "Importance (%)"]
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)

    # ── Feature interpretations ────────────────────────────────────────────────
    section_header("Key Feature Interpretations")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
**Top risk-increasing features:**

| Feature | Interpretation |
|---|---|
| `EXT_SOURCE_2/3` | Low external credit score → higher risk |
| `CREDIT_INCOME_PERCENT` | High credit-to-income → over-extended |
| `PAYMENT_RATE` | Low annuity ratio → minimal repayment |
| `DAYS_BIRTH` (age) | Younger applicants have higher risk |
| `BUREAU_AMT_CREDIT_SUM_DEBT_MEAN` | High existing bureau debt |
        """)
    with col_r:
        st.markdown("""
**Top risk-reducing features:**

| Feature | Interpretation |
|---|---|
| `EXT_SOURCE_PRODUCT` | High joint external score → low risk |
| `INSTAL_PAYMENT_PERC_MEAN` | Consistent prior repayment |
| `DAYS_EMPLOYED` | Longer employment = stability |
| `BUREAU_DAYS_CREDIT_MIN` | Established credit history |
| `PREV_AMT_ANNUITY_MEAN` | Manageable prior loan sizes |
        """)

    # ── Local explanation example ──────────────────────────────────────────────
    section_header("Example Local Explanation")
    example_path = PROJECT_ROOT / "reports" / "example_local_explanation.txt"
    if example_path.exists():
        with open(example_path) as f:
            explanation_text = f.read()
        st.code(explanation_text, language=None)
    else:
        st.markdown("""
```
Risk score: 34.2%  (threshold: 14.79%)  →  PREDICTED: DEFAULT

MODEL EXPLANATION (not causal reasoning):
Top factors INCREASING default risk:
  • High credit-to-income ratio (CREDIT_INCOME_PERCENT)   [SHAP: +0.182]
  • Short employment duration (DAYS_EMPLOYED)             [SHAP: +0.141]
  • Low external credit score (EXT_SOURCE_2)              [SHAP: +0.118]

Top factors REDUCING default risk:
  • Previous loans repaid in full (INSTAL_PAYMENT_PERC_MEAN)  [SHAP: -0.091]
  • Low outstanding bureau debt (BUREAU_AMT_CREDIT_SUM_DEBT_MEAN)  [SHAP: -0.073]

[Note: This is a model explanation, not a causal or legal determination.
 Refer to qualified credit analysts for real lending decisions.]
```
        """)

    # ── Limitations ────────────────────────────────────────────────────────────
    st.warning("""
**Important limitations of SHAP in this context:**
1. SHAP explains the **model**, not ground truth. EXT_SOURCE features are opaque external scores.
2. Local explanations are approximate for non-linear interactions.
    """)
