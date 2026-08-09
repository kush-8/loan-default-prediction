"""
evaluation.py
=============
Standalone evaluation utilities for the loan default prediction model.

Provides
--------
- Full metrics suite (ROC-AUC, PR-AUC, Brier, calibration curve, etc.)
- Threshold comparison table
- Model comparison across multiple classifiers
- Calibration diagnostics
- Pretty-print helpers

This module is imported by train.py and can also be run standalone
against a saved pipeline + test dataset.

Run
---
    python src/evaluation.py
"""

import json
import logging
import os
import sys
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def full_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    label: str = "",
) -> dict[str, float]:
    """
    Compute a comprehensive set of evaluation metrics.

    Parameters
    ----------
    y_true : array-like of 0/1
    y_proba : array-like of predicted probabilities for the positive class
    threshold : classification threshold
    label : optional prefix label for logging

    Returns
    -------
    dict with all metrics
    """
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "log_loss": log_loss(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "npv": npv,
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "threshold_used": threshold,
        "support_positive": int(y_true.sum()),
        "support_total": len(y_true),
    }

    if label:
        prefix = f"[{label}] "
        logger.info(
            f"{prefix}ROC-AUC={metrics['roc_auc']:.4f}  "
            f"PR-AUC={metrics['pr_auc']:.4f}  "
            f"Brier={metrics['brier_score']:.4f}"
        )
        logger.info(
            f"{prefix}Precision={metrics['precision']:.4f}  "
            f"Recall={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"@threshold={threshold:.3f}"
        )
        logger.info(f"{prefix}TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        logger.info(
            f"{prefix}NOTE: FP = legitimate borrowers wrongly flagged (lost revenue). "
            f"FN = defaulters missed (credit loss)."
        )

    return {k: float(v) if not isinstance(v, int) else v for k, v in metrics.items()}


# ---------------------------------------------------------------------------
# Threshold analysis
# ---------------------------------------------------------------------------


def threshold_comparison_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold_candidates: dict[str, float],
) -> pd.DataFrame:
    """
    Build a comparison table across multiple threshold strategies.

    Parameters
    ----------
    y_true : array-like of 0/1
    y_proba : predicted probabilities
    threshold_candidates : dict of {strategy_name: threshold_value}

    Returns
    -------
    pd.DataFrame with one row per strategy
    """
    rows = []
    for name, t in threshold_candidates.items():
        m = full_metrics(y_true, y_proba, threshold=t)
        rows.append(
            {
                "Strategy": name,
                "Threshold": round(t, 3),
                "Precision": round(m["precision"], 3),
                "Recall": round(m["recall"], 3),
                "Specificity": round(m["specificity"], 3),
                "F1": round(m["f1"], 3),
                "FP": m["false_positives"],
                "FN": m["false_negatives"],
                "Brier": round(m["brier_score"], 4),
            }
        )
    return pd.DataFrame(rows).set_index("Strategy")


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_report(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    label: str = "Model",
    save_path: str | None = None,
) -> dict[str, Any]:
    """
    Compute and optionally plot the calibration curve.

    Returns dict with ECE (expected calibration error) and Brier score.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)

    # Expected Calibration Error
    # Weight each bin by the number of samples
    counts = np.histogram(y_proba, bins=n_bins, range=(0, 1))[0]
    ece = float(np.average(np.abs(prob_true - prob_pred), weights=counts + 1e-8))
    brier = float(brier_score_loss(y_true, y_proba))

    if save_path:
        _fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.6)
        ax.plot(prob_pred, prob_true, "s-", label=label, color="steelblue")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"Calibration Curve — {label}\nBrier={brier:.4f}  ECE={ece:.4f}")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        logger.info(f"Calibration curve saved to {save_path}")

    return {"brier_score": brier, "ece": ece}


# ---------------------------------------------------------------------------
# ROC / PR curves
# ---------------------------------------------------------------------------


def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    label: str = "LightGBM",
    save_dir: str = "reports",
) -> None:
    """Save ROC and Precision-Recall curve plots."""
    os.makedirs(save_dir, exist_ok=True)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    _fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.4f})", color="steelblue")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(save_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=150)
    plt.close()

    # PR curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    _fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"{label} (AP={pr_auc:.4f})", color="darkorange")
    baseline = float(y_true.mean())
    ax.axhline(
        baseline,
        color="k",
        linestyle="--",
        alpha=0.5,
        label=f"Baseline (={baseline:.3f})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(save_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=150)
    plt.close()
    logger.info(f"ROC curve → {roc_path}  PR curve → {pr_path}")


# ---------------------------------------------------------------------------
# Model comparison (baseline benchmarking)
# ---------------------------------------------------------------------------


def benchmark_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compare Logistic Regression, Random Forest, and LightGBM on the same split.

    All models receive the same pre-processed feature matrix.
    No hyperparameter tuning is performed here — this establishes baseline
    performance. LightGBM is expected to outperform but the comparison
    documents the magnitude of improvement.

    Returns
    -------
    pd.DataFrame — comparison table
    """
    import time

    import lightgbm as lgb

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=seed, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=seed, class_weight="balanced", n_jobs=-1
        ),
        "LightGBM": lgb.LGBMClassifier(n_estimators=300, random_state=seed, verbose=-1),
    }

    rows = []
    for name, model in models.items():
        logger.info(f"  Benchmarking {name}...")
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t1 = time.time()
        proba = model.predict_proba(X_val)[:, 1]
        inf_time_ms = (time.time() - t1) * 1000 / len(X_val)

        m = full_metrics(y_val, proba, threshold=0.5)
        rows.append(
            {
                "Model": name,
                "ROC-AUC": round(m["roc_auc"], 4),
                "PR-AUC": round(m["pr_auc"], 4),
                "Recall@0.5": round(m["recall"], 4),
                "Precision@0.5": round(m["precision"], 4),
                "F1@0.5": round(m["f1"], 4),
                "Brier": round(m["brier_score"], 4),
                "Train time (s)": round(train_time, 1),
                "Inf time (ms/sample)": round(inf_time_ms, 3),
            }
        )

    df = pd.DataFrame(rows).set_index("Model")
    logger.info("\nModel Comparison Table:\n" + df.to_string())
    return df


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    pipeline_path = config["model_assets"]["pipeline_path"]
    if not os.path.exists(pipeline_path):
        logger.error(f"No pipeline found at {pipeline_path}. Run train.py first.")
        sys.exit(1)

    pipeline = joblib.load(pipeline_path)
    sample_df = pd.read_csv(config["data_paths"]["test_sample"])

    if "TARGET" not in sample_df.columns:
        logger.error("test_sample.csv must contain TARGET column for evaluation.")
        sys.exit(1)

    X_sample = sample_df.drop(columns=["TARGET"])
    y_sample = sample_df["TARGET"].values

    proba = pipeline.predict_proba(X_sample)[:, 1]

    # Load metadata for threshold
    metadata_path = os.path.join(os.path.dirname(pipeline_path), "model_metadata.json")
    threshold = 0.5
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        threshold = meta.get("threshold", {}).get("selected", 0.5)

    logger.info(f"Using threshold from metadata: {threshold}")
    metrics = full_metrics(y_sample, proba, threshold=threshold, label="Sample eval")

    os.makedirs("reports", exist_ok=True)
    calibration_report(y_sample, proba, label="LightGBM", save_path="reports/calibration_curve.png")
    plot_roc_pr_curves(y_sample, proba, save_dir="reports")
    logger.info("Evaluation complete. Plots saved to reports/")
