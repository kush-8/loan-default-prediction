"""
explain.py
==========
Batch SHAP explainability report generator.

Generates and saves:
- Global feature importance plot (SHAP summary)
- Calibration curve
- Reliability diagram
- Local explanation for one sample applicant (HTML force plot)
- Text explanation example

Run
---
    python src/explain.py
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path so joblib can unpickle src.models.preprocessing
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature name mapping (same as in predict.py — avoids circular imports)
# ---------------------------------------------------------------------------

_FEATURE_DISPLAY_NAMES = {
    "PAYMENT_RATE": "Payment Rate (Annuity/Credit)",
    "EXT_SOURCE_2": "External Credit Score 2",
    "EXT_SOURCE_3": "External Credit Score 3",
    "EXT_SOURCE_PRODUCT": "Combined External Score",
    "DAYS_EMPLOYED": "Employment Duration",
    "DAYS_BIRTH": "Applicant Age (Days)",
    "YEARS_BIRTH": "Applicant Age (Years)",
    "ANNUITY_INCOME_PERCENT": "Annuity / Income Ratio",
    "CREDIT_INCOME_PERCENT": "Credit / Income Ratio",
    "AMT_CREDIT": "Credit Amount",
    "AMT_INCOME_TOTAL": "Annual Income",
    "AMT_ANNUITY": "Loan Annuity",
    "AMT_GOODS_PRICE": "Goods Price",
    "DAYS_ID_PUBLISH": "Days Since ID Issued",
    "DAYS_REGISTRATION": "Days Since Registration",
    "DAYS_LAST_PHONE_CHANGE": "Days Since Phone Change",
    "REGION_POPULATION_RELATIVE": "Region Population Density",
    "BUREAU_DAYS_CREDIT_MAX": "Most Recent Bureau Enquiry",
    "BUREAU_DAYS_CREDIT_MIN": "Oldest Bureau Enquiry",
    "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN": "Avg Outstanding Debt (Bureau)",
    "BUREAU_AMT_CREDIT_SUM_MEAN": "Avg Bureau Credit Limit",
    "INSTAL_PAYMENT_PERC_MEAN": "Avg Instalment Payment %",
    "INSTAL_PAYMENT_DIFF_MEAN": "Avg Instalment Shortfall",
    "POS_SK_DPD_MAX": "Max Days Past Due (POS)",
    "CODE_GENDER_M": "Gender: Male",
    "FLAG_OWN_CAR": "Owns Car",
    "TOTALAREA_MODE": "Property Area",
}


def _readable_name(raw_name: str) -> str:
    clean = raw_name.split("__")[-1] if "__" in raw_name else raw_name
    return _FEATURE_DISPLAY_NAMES.get(clean, clean.replace("_", " ").title())


def _readable_feature_names(raw_names) -> list:
    return [_readable_name(n) for n in raw_names]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def generate_explanations(config_path: str = "config/config.yaml") -> None:
    logger.info("--- Generating model explanations ---")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    pipeline_path = config["model_assets"]["pipeline_path"]
    if not os.path.exists(pipeline_path):
        logger.error(f"Pipeline not found at {pipeline_path}. Run train.py first.")
        return

    pipeline = joblib.load(pipeline_path)
    logger.info("Pipeline loaded.")

    # Load a representative sample (use test_sample.csv if available)
    sample_path = config["data_paths"].get("test_sample")
    if sample_path and os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        X_sample = df.drop(columns=["TARGET"], errors="ignore").sample(
            n=min(200, len(df)), random_state=42
        )
        logger.info(f"Using test_sample.csv: {len(X_sample)} rows")
    else:
        train_path = config["data_paths"]["application_train"]
        df = pd.read_csv(train_path)
        X_sample = df.drop(columns=["TARGET"]).sample(n=200, random_state=42)
        logger.info("Using application_train.csv sample: 200 rows")

    # Extract pipeline steps
    fe = pipeline.named_steps.get("feature_engineering")
    preprocessor = pipeline.named_steps.get("preprocessor")
    classifier = pipeline.named_steps.get("classifier")

    if any(s is None for s in [fe, preprocessor, classifier]):
        logger.error(
            "Pipeline steps not found (expected: feature_engineering, preprocessor, classifier)."
        )
        return

    # Transform sample
    logger.info("Applying feature engineering and preprocessing...")
    X_eng = fe.transform(X_sample)
    drop_cols = [c for c in ["SK_ID_CURR"] if c in X_eng.columns]
    X_pp = preprocessor.transform(X_eng.drop(columns=drop_cols, errors="ignore"))
    raw_feature_names = preprocessor.get_feature_names_out()
    readable_names = _readable_feature_names(raw_feature_names)

    X_pp_df = pd.DataFrame(X_pp, columns=readable_names)

    # Unwrap calibrated classifier if needed
    base_clf = classifier
    if hasattr(classifier, "base_estimator"):
        base_clf = classifier.base_estimator
    elif hasattr(classifier, "estimator"):
        base_clf = classifier.estimator

    # SHAP values
    logger.info("Computing SHAP values (this may take a minute)...")
    explainer = shap.TreeExplainer(base_clf)
    shap_values = explainer(X_pp_df)

    os.makedirs("reports", exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Global summary plot
    # ------------------------------------------------------------------
    logger.info("Saving global SHAP summary plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_pp_df, show=False, max_display=20)
    plt.title(
        "Global Feature Importance (SHAP)\n"
        "Each point = one sample. Color = feature value. "
        "x-axis = impact on default probability.",
        fontsize=9,
    )
    plt.tight_layout()
    plt.savefig("reports/shap_summary_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: reports/shap_summary_plot.png")

    # ------------------------------------------------------------------
    # 2. Bar plot of mean |SHAP| values
    # ------------------------------------------------------------------
    logger.info("Saving mean SHAP importance bar plot...")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_pp_df, plot_type="bar", show=False, max_display=20)
    plt.title("Mean |SHAP Value| — Global Feature Importance", fontsize=10)
    plt.tight_layout()
    plt.savefig("reports/shap_bar_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved: reports/shap_bar_plot.png")

    # ------------------------------------------------------------------
    # 3. Local force plot (HTML) — single applicant
    # ------------------------------------------------------------------
    logger.info("Saving local SHAP force plot (sample 0)...")
    try:
        force_plot_html = shap.force_plot(
            shap_values.base_values[0],
            shap_values.values[0],
            X_pp_df.iloc[0],
            matplotlib=False,
        )
        shap.save_html("reports/shap_force_plot_single.html", force_plot_html)
        logger.info("Saved: reports/shap_force_plot_single.html")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Force plot failed (harmless): {exc}")

    # ------------------------------------------------------------------
    # 4. Text explanation example
    # ------------------------------------------------------------------
    logger.info("Generating text explanation example...")
    sv_single = shap_values.values[0]
    indices = np.argsort(np.abs(sv_single))[::-1]

    risk_lines = []
    protect_lines = []
    for idx in indices:
        label = readable_names[idx] if idx < len(readable_names) else f"feature_{idx}"
        val = sv_single[idx]
        if val > 0 and len(risk_lines) < 5:
            risk_lines.append(f"  ↑  {label} (impact: +{val:.4f})")
        elif val < 0 and len(protect_lines) < 5:
            protect_lines.append(f"  ↓  {label} (impact: {val:.4f})")
        if len(risk_lines) >= 5 and len(protect_lines) >= 5:
            break

    proba_sample = pipeline.predict_proba(X_sample.iloc[:1])[:, 1][0]
    text_example = (
        f"=== Local Prediction Explanation (Sample 0) ===\n"
        f"Risk score: {proba_sample * 100:.1f}%\n\n"
        f"Main factors INCREASING default risk:\n"
        + "\n".join(risk_lines)
        + "\n\nMain factors REDUCING default risk:\n"
        + "\n".join(protect_lines)
        + "\n\n[Note: This is a model explanation, not a causal or legal determination.]"
    )
    try:
        print(text_example)
    except UnicodeEncodeError:
        print(text_example.encode("utf-8", errors="replace").decode("utf-8", errors="replace"))

    with open("reports/example_local_explanation.txt", "w", encoding="utf-8") as f:
        f.write(text_example)
    logger.info("Saved: reports/example_local_explanation.txt")
    logger.info("--- Explanation generation complete ---")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    os.makedirs("reports", exist_ok=True)
    generate_explanations(config_path=cfg)
