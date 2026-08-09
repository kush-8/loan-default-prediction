"""
predict.py
==========
Model loading and inference utilities.

The ``ModelRegistry`` class loads the pipeline and metadata ONCE and
provides a ``predict()`` method used by the API. It is not tied to FastAPI
so it can be used in batch scripts and tests without importing the web framework.

Batch prediction
----------------
    python src/predict.py                        # uses config/config.yaml
    python src/predict.py config/config.yaml     # explicit config path
"""

import json
import logging
import os
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Human-readable feature name mapping
# ---------------------------------------------------------------------------

_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "PAYMENT_RATE": "annuity-to-credit ratio (payment rate)",
    "EXT_SOURCE_2": "external credit score 2",
    "EXT_SOURCE_3": "external credit score 3",
    "EXT_SOURCE_PRODUCT": "combined external credit score",
    "DAYS_EMPLOYED": "employment duration (days)",
    "DAYS_BIRTH": "applicant age (days)",
    "YEARS_BIRTH": "applicant age (years)",
    "ANNUITY_INCOME_PERCENT": "annuity as % of income",
    "CREDIT_INCOME_PERCENT": "credit amount as % of income",
    "AMT_CREDIT": "requested credit amount",
    "AMT_INCOME_TOTAL": "declared annual income",
    "AMT_ANNUITY": "loan annuity amount",
    "AMT_GOODS_PRICE": "goods price",
    "DAYS_ID_PUBLISH": "days since ID was issued",
    "DAYS_REGISTRATION": "days since address registration",
    "DAYS_LAST_PHONE_CHANGE": "days since last phone change",
    "REGION_POPULATION_RELATIVE": "region population density",
    "BUREAU_DAYS_CREDIT_MAX": "most recent credit bureau enquiry (days)",
    "BUREAU_DAYS_CREDIT_MIN": "oldest credit bureau enquiry (days)",
    "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN": "average outstanding bureau debt",
    "BUREAU_AMT_CREDIT_SUM_MEAN": "average bureau credit limit",
    "INSTAL_PAYMENT_PERC_MEAN": "average instalment payment percentage",
    "INSTAL_PAYMENT_DIFF_MEAN": "average instalment shortfall",
    "POS_SK_DPD_MAX": "max days-past-due (POS loans)",
    "POS_CNT_INSTALMENT_FUTURE_MEAN": "average remaining instalments",
    "CODE_GENDER_M": "gender (male)",
    "NAME_EDUCATION_TYPE_ENCODED": "education level",
    "FLAG_OWN_CAR": "car ownership flag",
    "NAME_FAMILY_STATUS_Married": "marital status (married)",
    "TOTALAREA_MODE": "apartment area mode",
}


def _readable_name(raw_name: str) -> str:
    """Return a human-readable label for a feature name."""
    # Strip sklearn ColumnTransformer prefixes (e.g. "num__PAYMENT_RATE")
    clean = raw_name.split("__")[-1] if "__" in raw_name else raw_name
    return _FEATURE_DISPLAY_NAMES.get(clean, clean.replace("_", " ").title())


# ---------------------------------------------------------------------------
# SHAP-based local explanation
# ---------------------------------------------------------------------------


def _local_explanation(
    pipeline,
    input_df: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, Any]:
    """
    Compute a SHAP-based local explanation for a single prediction.

    Returns a dict with:
    - top_risk_factors: list of (feature, direction, human_label)
    - top_protective_factors: list of (feature, direction, human_label)
    - text_summary: human-readable text

    Falls back gracefully if SHAP is unavailable or computation fails.
    """
    try:
        import shap

        fe = pipeline.named_steps.get("feature_engineering")
        preprocessor = pipeline.named_steps.get("preprocessor")
        classifier = pipeline.named_steps.get("classifier")

        if fe is None or preprocessor is None or classifier is None:
            return {}

        # Transform the input
        input_eng = fe.transform(input_df)
        drop_cols = [c for c in ["SK_ID_CURR"] if c in input_eng.columns]
        input_pp = preprocessor.transform(
            input_eng.drop(columns=drop_cols, errors="ignore")
        )
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(
            classifier
            if not hasattr(classifier, "base_estimator")
            else classifier.base_estimator,
            feature_perturbation="interventional",
        )
        shap_values = explainer.shap_values(input_pp)

        # Handle binary classification (list of two arrays)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
        sv = sv[0]  # single sample

        # Sort features by |SHAP value|
        indices = np.argsort(np.abs(sv))[::-1]

        risk_factors = []
        protective_factors = []

        for idx in indices:
            if len(risk_factors) >= top_n and len(protective_factors) >= top_n:
                break
            feat_name = (
                feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
            )
            shap_val = float(sv[idx])
            label = _readable_name(feat_name)

            if shap_val > 0 and len(risk_factors) < top_n:
                risk_factors.append(
                    {
                        "feature": feat_name,
                        "label": label,
                        "shap_value": round(shap_val, 4),
                    }
                )
            elif shap_val < 0 and len(protective_factors) < top_n:
                protective_factors.append(
                    {
                        "feature": feat_name,
                        "label": label,
                        "shap_value": round(shap_val, 4),
                    }
                )

        risk_lines = "\n".join(f"  • {f['label']}" for f in risk_factors)
        protect_lines = "\n".join(f"  • {f['label']}" for f in protective_factors)
        text_summary = (
            "MODEL EXPLANATION (not causal reasoning):\n"
            f"Top factors increasing default risk:\n{risk_lines or '  None'}\n"
            f"Top factors reducing default risk:\n{protect_lines or '  None'}"
        )

        return {
            "top_risk_factors": risk_factors,
            "top_protective_factors": protective_factors,
            "text_summary": text_summary,
        }

    except Exception as exc:  # noqa: BLE001
        logger.warning(f"SHAP explanation skipped: {exc}")
        return {}


# ---------------------------------------------------------------------------
# ModelRegistry — loads once, reused per request
# ---------------------------------------------------------------------------


class ModelRegistry:
    """
    Holds the loaded pipeline and metadata.

    Designed to be instantiated once at application startup and then
    shared across all requests via app.state.registry.
    """

    def __init__(self, pipeline, metadata: dict[str, Any]) -> None:
        self.pipeline = pipeline
        self.metadata = metadata
        self.threshold: float = metadata.get("threshold", {}).get("selected", 0.5)
        self.model_version: str = metadata.get("model_version", "unknown")

    @classmethod
    def from_config(cls, config_path: str = "config/config.yaml") -> "ModelRegistry":
        """Load pipeline and metadata from paths in config."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        pipeline_path = config["model_assets"]["pipeline_path"]
        if not os.path.exists(pipeline_path):
            raise FileNotFoundError(
                f"Model pipeline not found at '{pipeline_path}'. "
                "Run train.py to train and save the model."
            )

        logger.info(f"Loading pipeline from {pipeline_path}...")
        pipeline = joblib.load(pipeline_path)

        # Load metadata (optional)
        metadata_path = os.path.join(
            os.path.dirname(pipeline_path), "model_metadata.json"
        )
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                metadata = json.load(f)
            logger.info(
                f"Metadata loaded. Version: {metadata.get('model_version', 'unknown')}"
            )
        else:
            logger.warning(f"No metadata file at {metadata_path}. Using defaults.")
            metadata = {}

        return cls(pipeline=pipeline, metadata=metadata)

    def predict(
        self,
        input_df: pd.DataFrame,
        threshold: float | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """
        Make a prediction for a DataFrame with one or more rows.

        Parameters
        ----------
        input_df : pd.DataFrame
            Raw application table features (same schema as training input).
        threshold : float, optional
            Override the stored threshold.
        explain : bool
            If True, compute SHAP-based local explanation (single rows only).

        Returns
        -------
        dict with prediction results
        """
        t = threshold if threshold is not None else self.threshold

        probabilities = self.pipeline.predict_proba(input_df)[:, 1]

        results = {
            "probability": float(round(probabilities[0], 6))
            if len(probabilities) == 1
            else [float(round(p, 6)) for p in probabilities],
            "predicted_class": int(probabilities[0] >= t)
            if len(probabilities) == 1
            else [int(p >= t) for p in probabilities],
            "threshold": float(t),
            "model_version": self.model_version,
        }

        if explain and len(probabilities) == 1:
            explanation = _local_explanation(self.pipeline, input_df)
            results["explanation"] = explanation

        return results


# ---------------------------------------------------------------------------
# Standalone batch prediction
# ---------------------------------------------------------------------------


def run_batch_prediction(config_path: str = "config/config.yaml") -> None:
    """Run predictions on application_test.csv and save results."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    registry = ModelRegistry.from_config(config_path)

    test_path = config["data_paths"]["application_test"]
    logger.info(f"Loading test data from {test_path}")
    test_df = pd.read_csv(test_path)

    logger.info(f"Running batch predictions on {len(test_df):,} rows...")
    probabilities = registry.pipeline.predict_proba(test_df)[:, 1]

    result_df = pd.DataFrame(
        {"SK_ID_CURR": test_df["SK_ID_CURR"], "TARGET": probabilities}
    )

    out_path = config["data_paths"]["test_result"]
    result_df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(result_df):,} predictions to {out_path}")
    logger.info(
        f"Prediction distribution: mean={probabilities.mean():.4f}  "
        f"p50={np.median(probabilities):.4f}  "
        f"p90={np.percentile(probabilities, 90):.4f}"
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    run_batch_prediction(config_path=cfg)
