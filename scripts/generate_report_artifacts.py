"""
generate_report_artifacts.py
=============================
Generate JSON evaluation artifacts for the Streamlit dashboard.

Loads the saved sklearn pipeline + test_sample.csv and computes:
- ROC curve data points
- PR curve data points
- Calibration curve data
- Threshold comparison table (precision/recall/F1 at each strategy threshold)
- Model benchmark table
- Top feature importance (from LightGBM feature_importances_ or SHAP mean |values|)

Output: reports/evaluation_artifacts.json

Usage
-----
    python scripts/generate_report_artifacts.py

Requirements
------------
- models/final_pipeline.joblib must exist (run train.py first)
- data/processed/test_sample.csv must exist
"""

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_metadata() -> dict:
    with open(PROJECT_ROOT / "models" / "model_metadata.json") as f:
        return json.load(f)


def _numpy_safe(obj):
    """Recursively convert numpy types to Python native for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _numpy_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_numpy_safe(v) for v in obj]
    return obj


def generate_roc_curve(y_true, y_score) -> dict:
    """Compute ROC curve with downsampled points for dashboard rendering."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    # Downsample to 200 points max for efficient rendering
    idx = np.linspace(0, len(fpr) - 1, min(200, len(fpr))).astype(int)
    return {
        "fpr": fpr[idx].tolist(),
        "tpr": tpr[idx].tolist(),
        "auc": round(float(auc), 4),
    }


def generate_pr_curve(y_true, y_score) -> dict:
    """Compute precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    auc = average_precision_score(y_true, y_score)

    # Downsample
    idx = np.linspace(0, len(precision) - 1, min(200, len(precision))).astype(int)
    return {
        "precision": precision[idx].tolist(),
        "recall": recall[idx].tolist(),
        "auc": round(float(auc), 4),
        "baseline": round(float(y_true.mean()), 4),  # random classifier baseline
    }


def generate_calibration_curve(y_true, y_score, n_bins: int = 10) -> dict:
    """Compute calibration curve (reliability diagram data)."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="uniform"
    )
    brier = brier_score_loss(y_true, y_score)
    return {
        "mean_predicted": mean_predicted_value.tolist(),
        "fraction_positive": fraction_of_positives.tolist(),
        "brier_score": round(float(brier), 6),
        "n_bins": n_bins,
    }


def generate_threshold_comparison(y_true, y_score, threshold_candidates: dict) -> list[dict]:
    """Compute precision/recall/F1 at each threshold candidate."""
    rows = []
    for strategy, thresh in threshold_candidates.items():
        preds = (y_score >= thresh).astype(int)
        rows.append({
            "strategy": strategy,
            "threshold": round(float(thresh), 4),
            "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
            "predicted_positive_rate": round(float(preds.mean()), 4),
            "selected": strategy == "f1_optimal",
        })
    return rows


def generate_feature_importance(pipeline) -> list[dict]:
    """
    Extract feature importances from the fitted LightGBM classifier.
    Returns top 30 features sorted by mean gain importance.
    """
    clf = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        booster = clf.booster_
        importance = booster.feature_importance(importance_type="gain")
    except Exception:
        logger.warning("Could not extract LightGBM booster — falling back to feature_importances_")
        importance = clf.feature_importances_

    try:
        # Extract true feature names from the ColumnTransformer
        raw_names = preprocessor.get_feature_names_out()
        # Clean up the names (e.g. 'num__EXT_SOURCE_2' -> 'EXT_SOURCE_2')
        feature_names = [name.split("__")[-1] if "__" in name else name for name in raw_names]
    except Exception:
        logger.warning("Could not extract feature names from preprocessor")
        feature_names = [f"feature_{i}" for i in range(len(importance))]

    total = importance.sum()
    if total > 0:
        importance_norm = importance / total
    else:
        importance_norm = importance

    # Top 30 features
    top_idx = np.argsort(importance_norm)[::-1][:30]
    result = []
    for i in top_idx:
        result.append({
            "feature": str(feature_names[i]),
            "importance": round(float(importance_norm[i]), 6),
            "importance_raw": round(float(importance[i]), 2),
        })
    return result


def generate_model_benchmark() -> list[dict]:
    """
    Return the model comparison benchmark table.
    Values documented in experiment_report.md Experiment 4.1.
    """
    return [
        {
            "model": "Logistic Regression",
            "roc_auc": 0.67,
            "pr_auc": 0.22,
            "brier_score": 0.073,
            "train_time_s": 5,
            "inference_ms": 0.1,
            "selected": False,
        },
        {
            "model": "Random Forest",
            "roc_auc": 0.72,
            "pr_auc": 0.31,
            "brier_score": 0.069,
            "train_time_s": 180,
            "inference_ms": 0.5,
            "selected": False,
        },
        {
            "model": "LightGBM (Optuna-tuned)",
            "roc_auc": 0.7720,
            "pr_auc": 0.2561,
            "brier_score": 0.0671,
            "train_time_s": 67,
            "inference_ms": 0.1,
            "selected": True,
        },
    ]


def main():
    config = load_config()
    metadata = load_metadata()

    # Load pipeline
    pipeline_path = PROJECT_ROOT / config["model_assets"]["pipeline_path"]
    if not pipeline_path.exists():
        logger.error(f"Pipeline not found at {pipeline_path}. Run train.py first.")
        sys.exit(1)

    logger.info(f"Loading pipeline from {pipeline_path}...")
    pipeline = joblib.load(pipeline_path)

    # Load test sample
    sample_path = PROJECT_ROOT / config["data_paths"]["test_sample"]
    if not sample_path.exists():
        logger.error(f"Test sample not found at {sample_path}.")
        sys.exit(1)

    logger.info(f"Loading test sample from {sample_path}...")
    sample_df = pd.read_csv(sample_path)
    X_sample = sample_df.drop(columns=["TARGET"], errors="ignore")
    y_sample = sample_df["TARGET"].values

    logger.info(f"Generating predictions on {len(X_sample)} test samples...")
    y_score = pipeline.predict_proba(X_sample)[:, 1]

    threshold_candidates = metadata["threshold"]["all_candidates"]

    logger.info("Computing ROC curve...")
    roc = generate_roc_curve(y_sample, y_score)

    logger.info("Computing PR curve...")
    pr = generate_pr_curve(y_sample, y_score)

    logger.info("Computing calibration curve...")
    cal = generate_calibration_curve(y_sample, y_score)

    logger.info("Computing threshold comparison...")
    thresh_comparison = generate_threshold_comparison(y_sample, y_score, threshold_candidates)

    logger.info("Extracting feature importance...")
    feat_importance = generate_feature_importance(pipeline)

    logger.info("Building benchmark table...")
    benchmark = generate_model_benchmark()

    # Assemble artifacts
    artifacts = {
        "_meta": {
            "generated_by": "scripts/generate_report_artifacts.py",
            "model_version": metadata["model_version"],
            "test_sample_n": int(len(X_sample)),
            "positive_rate": round(float(y_sample.mean()), 4),
        },
        "roc_curve": roc,
        "pr_curve": pr,
        "calibration_curve": cal,
        "threshold_comparison": thresh_comparison,
        "feature_importance": feat_importance,
        "model_benchmark": benchmark,
        "test_metrics": metadata["test_metrics"],
        "validation_metrics": metadata["validation_metrics"],
        "threshold": metadata["threshold"],
        "model": metadata["model"],
        "training_timestamp": metadata["training_timestamp"],
        "feature_engineering": metadata["feature_engineering"],
    }

    output_path = PROJECT_ROOT / "reports" / "evaluation_artifacts.json"
    with open(output_path, "w") as f:
        json.dump(_numpy_safe(artifacts), f, indent=2)

    logger.info(f"Artifacts saved to {output_path}")
    logger.info("Summary:")
    logger.info(f"  ROC-AUC (test): {roc['auc']}")
    logger.info(f"  PR-AUC  (test): {pr['auc']}")
    logger.info(f"  Brier   (test): {cal['brier_score']}")
    logger.info(f"  Top feature: {feat_importance[0]['feature']} ({feat_importance[0]['importance']:.4f})")


if __name__ == "__main__":
    main()
