"""Page 10: Live Prediction — Form-based applicant scoring."""

import pandas as pd
import streamlit as st

from app.utils import (
    load_metadata,
    load_pipeline,
    section_header,
)


# ── Risk category helper ───────────────────────────────────────────────────────
def _risk_badge(prob: float) -> tuple[str, str]:
    if prob < 0.15:
        return "LOW", "risk-low"
    elif prob < 0.30:
        return "MEDIUM", "risk-medium"
    elif prob < 0.50:
        return "HIGH", "risk-high"
    return "VERY HIGH", "risk-vhigh"


def render():
    st.markdown("# 🔮 Live Prediction")
    st.markdown("Fill in applicant features to see a model prediction. ")

    meta = load_metadata()
    pipeline = load_pipeline()
    threshold = meta.get("threshold", {}).get("selected", 0.1479)
    model_version = meta.get("model_version", "1.1.0")

    if pipeline is None:
        st.error(
            "Model pipeline not found. Ensure `models/final_pipeline.joblib` exists. "
            "Run `python scripts/download_and_train.py` to train the model."
        )
        return

    # ── Input form ─────────────────────────────────────────────────────────────
    section_header("Applicant Information")
    st.caption(
        "Enter approximate values. Fields left at default are imputed by the model pipeline."
    )

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Credit Details**")
            amt_credit = st.number_input(
                "Credit Amount (USD)", min_value=0.0, value=270000.0, step=5000.0
            )
            amt_income = st.number_input(
                "Annual Income (USD)", min_value=0.0, value=135000.0, step=5000.0
            )
            amt_annuity = st.number_input(
                "Loan Annuity (USD/mo)", min_value=0.0, value=13500.0, step=500.0
            )
            amt_goods = st.number_input(
                "Goods Price (USD)", min_value=0.0, value=225000.0, step=5000.0
            )

        with col2:
            st.markdown("**Personal Details**")
            age_years = st.slider("Applicant Age (years)", min_value=18, max_value=70, value=35)
            gender = st.selectbox("Gender", ["M", "F"])
            education = st.selectbox(
                "Education",
                [
                    "Secondary / secondary special",
                    "Higher education",
                    "Incomplete higher",
                    "Lower secondary",
                    "Academic degree",
                ],
            )
            family_status = st.selectbox(
                "Family Status",
                [
                    "Married",
                    "Single / not married",
                    "Civil marriage",
                    "Separated",
                    "Widow",
                ],
            )
            own_car = st.checkbox("Owns a car", value=False)

        with col3:
            st.markdown("**Employment & Credit History**")
            days_employed = st.number_input(
                "Days employed (negative = months back)",
                min_value=-18000,
                max_value=0,
                value=-1500,
                step=30,
            )
            ext_source_2 = st.slider("External Credit Score 2", 0.0, 1.0, 0.55, 0.01)
            ext_source_3 = st.slider("External Credit Score 3", 0.0, 1.0, 0.55, 0.01)
            region_pop = st.number_input(
                "Region Population Relative",
                min_value=0.0,
                max_value=0.1,
                value=0.019,
                step=0.001,
                format="%.4f",
            )

        submitted = st.form_submit_button("🔮 Score Applicant", use_container_width=True)

    # ── Prediction ─────────────────────────────────────────────────────────────
    if submitted:
        # Build a minimal application row
        # We must supply all expected columns — use NaN for everything not entered
        # The pipeline's FullFeatureEngineering will compute ratio features internally
        input_data = {
            "SK_ID_CURR": [999999],
            "TARGET": [None],  # will be dropped
            "AMT_CREDIT": [float(amt_credit)],
            "AMT_INCOME_TOTAL": [float(amt_income)],
            "AMT_ANNUITY": [float(amt_annuity)],
            "AMT_GOODS_PRICE": [float(amt_goods)],
            "DAYS_BIRTH": [int(-age_years * 365)],
            "DAYS_EMPLOYED": [int(days_employed)],
            "CODE_GENDER": [gender],
            "NAME_EDUCATION_TYPE": [education],
            "NAME_FAMILY_STATUS": [family_status],
            "FLAG_OWN_CAR": ["Y" if own_car else "N"],
            "EXT_SOURCE_2": [float(ext_source_2)],
            "EXT_SOURCE_3": [float(ext_source_3)],
            "REGION_POPULATION_RELATIVE": [float(region_pop)],
        }
        input_df = pd.DataFrame(input_data)
        input_df = input_df.drop(columns=["TARGET"], errors="ignore")

        with st.spinner("Scoring applicant..."):
            try:
                prob = pipeline.predict_proba(input_df)[:, 1][0]
                predicted_class = int(prob >= threshold)
                label, css_class = _risk_badge(prob)

                # ── Result display ─────────────────────────────────────────────
                st.markdown("---")
                section_header("Prediction Result")

                c1, c2, c3 = st.columns([2, 2, 3])
                with c1:
                    st.metric("Default Probability", f"{prob:.2%}")
                with c2:
                    st.metric(
                        "Model Decision",
                        "⚠️ FLAG" if predicted_class == 1 else "✅ PASS",
                        f"threshold: {threshold:.4f}",
                    )
                with c3:
                    st.markdown(
                        f"<div style='text-align:center;padding-top:0.5rem'>"
                        f"<span class='risk-badge {css_class}'>{label} RISK</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # Risk gauge bar
                gauge_pct = int(prob * 100)
                bar_color = {
                    "LOW": "#10b981",
                    "MEDIUM": "#3b82f6",
                    "HIGH": "#f59e0b",
                    "VERY HIGH": "#ef4444",
                }[label]
                st.markdown(
                    f"""
                    <div style='margin: 1rem 0'>
                        <div style='display:flex;justify-content:space-between;font-size:0.75rem;color:#9ca3af;margin-bottom:4px'>
                            <span>0%</span><span>Threshold ({threshold:.0%})</span><span>100%</span>
                        </div>
                        <div style='background:#2d3147;border-radius:8px;height:24px;position:relative;overflow:hidden'>
                            <div style='width:{gauge_pct}%;background:{bar_color};height:100%;
                                border-radius:8px;transition:width 0.5s ease'></div>
                            <div style='position:absolute;left:{threshold*100:.0f}%;top:0;bottom:0;
                                width:2px;background:#e2e8f0;opacity:0.7'></div>
                        </div>
                        <div style='text-align:center;font-size:1.5rem;font-weight:800;
                            color:{bar_color};margin-top:4px'>{prob:.1%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Derived features
                with st.expander("Derived Features (computed by pipeline)", expanded=False):
                    payment_rate = amt_annuity / max(amt_credit, 1)
                    credit_income_pct = amt_credit / max(amt_income, 1)
                    annuity_income_pct = amt_annuity / max(amt_income, 1)
                    ext_product = ext_source_2 * ext_source_3

                    st.markdown(f"""
| Derived Feature | Value |
|---|---|
| `PAYMENT_RATE` | {payment_rate:.4f} |
| `CREDIT_INCOME_PERCENT` | {credit_income_pct:.4f} |
| `ANNUITY_INCOME_PERCENT` | {annuity_income_pct:.4f} |
| `EXT_SOURCE_PRODUCT` | {ext_product:.4f} |
| `YEARS_BIRTH` | {age_years} |
                    """)

                st.caption(
                    f"Model version: {model_version} | "
                    f"Threshold: {threshold:.4f} (F1-optimal on validation)"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.info(
                    "This may occur if the pipeline expects columns not present in the minimal input. "
                    "The full pipeline requires all 122 application table columns. "
                    "Partial inputs are filled with NaN and handled by the pipeline's imputer."
                )
