"""
train.py
========
End-to-end training pipeline for the loan default risk model.

Pipeline contract
-----------------
1. Raw application_train.csv  →  3-way stratified split (train / val / test)
2. Historical features built offline once, joined from cache.
3. FullFeatureEngineering  fitted on train, applied to val and test.
4. ColumnTransformer preprocessor fitted on train, applied to val and test.
5. LightGBM fitted on train.
6. Threshold selected on VALIDATION set (F1-optimal).
7. Calibration evaluated on VALIDATION set (CalibratedClassifierCV optional).
8. Final metrics reported on TEST set ONLY (never touched before this point).
9. Model pipeline + metadata saved to disk.

Reproducibility
---------------
- All random states use config['training_params']['random_state'].
- git commit SHA recorded in metadata when available.
- Dependency versions recorded in metadata.

Run
---
    python src/train.py
    python src/train.py config/config.yaml   # explicit config path
"""

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Project imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.preprocessing import FullFeatureEngineering, create_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    """Compute a comprehensive set of classification metrics."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "brier_score": float(brier_score_loss(y_true, y_proba)),
        "log_loss": float(log_loss(y_true, y_proba)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold_used": float(threshold),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    strategy: str = "f1",
    min_recall: float = 0.60,
    fp_cost: float = 1.0,
    fn_cost: float = 5.0,
) -> dict:
    """
    Find classification thresholds under multiple strategies.

    All thresholds are derived from validation data; the test set is NEVER
    used here.

    Strategies
    ----------
    f1       : maximise F1 score
    recall   : smallest threshold such that recall >= min_recall
    cost     : minimise fp_cost * FP + fn_cost * FN
    default  : 0.5 (baseline)

    Returns dict of {strategy_name: threshold}.
    """
    thresholds = np.linspace(0.01, 0.99, 200)
    results = {"default": 0.5}

    best_f1, best_f1_thresh = -1, 0.5
    best_cost, best_cost_thresh = np.inf, 0.5

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1, best_f1_thresh = f1, float(t)

        cost = fp_cost * fp + fn_cost * fn
        if cost < best_cost:
            best_cost, best_cost_thresh = cost, float(t)

    results["f1_optimal"] = best_f1_thresh
    results["cost_optimal"] = best_cost_thresh

    # Recall-constrained: smallest threshold satisfying recall >= min_recall
    # (lower threshold → more positives → higher recall)
    recall_thresh = 0.5
    for t in reversed(thresholds):  # iterate high → low threshold
        y_pred = (y_proba >= t).astype(int)
        r = recall_score(y_true, y_pred, zero_division=0)
        if r >= min_recall:
            recall_thresh = float(t)
            break
    results[f"recall_geq_{int(min_recall * 100)}pct"] = recall_thresh

    logger.info(f"Threshold candidates: {results}")
    return results


def get_git_sha() -> str | None:
    """Return the current git commit SHA, or None if not in a git repo."""
    try:
        sha = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .strip()
            .decode()
        )
        return sha
    except Exception:  # noqa: BLE001
        return None


def dataset_hash(path: str) -> str:
    """Compute MD5 hash of the first 10 MB of a CSV for lightweight fingerprinting."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        chunk = f.read(10 * 1024 * 1024)
        h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def run_training(config_path: str = "config/config.yaml") -> None:
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("Starting training pipeline")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load config
    # ------------------------------------------------------------------
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["training_params"]["random_state"]
    target_col = config["training_params"]["target_column"]

    # ------------------------------------------------------------------
    # 2. Load raw application data
    # ------------------------------------------------------------------
    train_path = config["data_paths"]["application_train"]
    logger.info(f"Loading training data from {train_path}")
    df = pd.read_csv(train_path)
    logger.info(f"Raw data shape: {df.shape}")

    # ------------------------------------------------------------------
    # 3. Three-way stratified split: train 70% / val 15% / test 15%
    # ------------------------------------------------------------------
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.15, random_state=seed, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=0.15 / 0.85,  # ~15% of total
        random_state=seed,
        stratify=y_trainval,
    )

    logger.info(
        f"Split sizes → train: {len(X_train):,}  val: {len(X_val):,}  " f"test: {len(X_test):,}"
    )
    logger.info(
        f"Positive rate → train: {y_train.mean():.3f}  val: {y_val.mean():.3f}  "
        f"test: {y_test.mean():.3f}"
    )

    # ------------------------------------------------------------------
    # 4. Feature engineering — fit ONLY on train
    # ------------------------------------------------------------------
    logger.info("Fitting FullFeatureEngineering on training data...")
    fe = FullFeatureEngineering(config_path=config_path)
    X_train_fe = fe.fit_transform(X_train)  # fit + transform

    logger.info("Transforming validation and test sets...")
    X_val_fe = fe.transform(X_val)
    X_test_fe = fe.transform(X_test)

    # ------------------------------------------------------------------
    # 5. Discover column types from training-engineered data
    # ------------------------------------------------------------------
    numerical_cols = [
        c for c in X_train_fe.select_dtypes(include=np.number).columns if c != "SK_ID_CURR"
    ]
    categorical_cols = X_train_fe.select_dtypes(include="object").columns.tolist()

    logger.info(
        f"Feature engineering output: {X_train_fe.shape[1]} cols "
        f"({len(numerical_cols)} numerical, {len(categorical_cols)} categorical)"
    )

    # ------------------------------------------------------------------
    # 6. Load hyperparameters
    # ------------------------------------------------------------------
    with open(config["model_assets"]["model_parameters"]) as f:
        best_params = json.load(f)

    # Remove sklearn-style aliases that conflict with LightGBM native params
    # (subsample/colsample_bytree are sklearn aliases for bagging_fraction/
    #  feature_fraction; using both causes a warning and may override values)
    for alias in ("subsample", "colsample_bytree"):
        if alias in best_params:
            logger.warning(
                f"Removing '{alias}' from best_params — use LightGBM native "
                f"'bagging_fraction'/'feature_fraction' instead."
            )
            best_params.pop(alias)

    best_params["random_state"] = seed
    best_params["verbose"] = -1  # suppress LightGBM stdout

    # ------------------------------------------------------------------
    # 7. Build and train the full pipeline
    # ------------------------------------------------------------------
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)

    # NOTE: We deliberately fit the preprocessor and classifier separately on
    # the pre-engineered data rather than assembling a single Pipeline here.
    # Reason: FullFeatureEngineering is already fitted (step 4 above). If we
    # called Pipeline.fit(), sklearn would call fe.fit_transform() again on the
    # training data — refitting bins and losing the fitted state. The inference
    # pipeline is assembled in step 11 below AFTER all components are fitted.

    logger.info("Training pipeline on training data...")
    # Train preprocessor and classifier on pre-engineered training data
    preprocessor.fit(X_train_fe.drop(columns=["SK_ID_CURR"]), y_train)
    X_train_pp = preprocessor.transform(X_train_fe.drop(columns=["SK_ID_CURR"]))
    X_val_pp = preprocessor.transform(X_val_fe.drop(columns=["SK_ID_CURR"]))
    X_test_pp = preprocessor.transform(X_test_fe.drop(columns=["SK_ID_CURR"]))

    classifier = lgb.LGBMClassifier(**best_params)
    logger.info("Fitting LightGBM classifier...")
    classifier.fit(
        X_train_pp,
        y_train,
        eval_set=[(X_val_pp, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    # ------------------------------------------------------------------
    # 8. Threshold selection — VALIDATION SET ONLY
    # ------------------------------------------------------------------
    logger.info("Selecting threshold on validation set...")
    val_proba = classifier.predict_proba(X_val_pp)[:, 1]
    threshold_candidates = find_optimal_threshold(y_val.values, val_proba)
    selected_threshold = threshold_candidates["f1_optimal"]
    logger.info(f"Selected threshold (F1-optimal on val): {selected_threshold:.4f}")

    # ------------------------------------------------------------------
    # 9. Probability calibration — VALIDATION SET ONLY
    # ------------------------------------------------------------------
    logger.info("Evaluating probability calibration on validation set...")

    # Baseline Brier score (uncalibrated)
    brier_uncal = brier_score_loss(y_val, val_proba)

    # Platt scaling (sigmoid)
    cal_sigmoid = CalibratedClassifierCV(classifier, cv="prefit", method="sigmoid")
    cal_sigmoid.fit(X_val_pp, y_val)
    val_proba_sig = cal_sigmoid.predict_proba(X_val_pp)[:, 1]
    brier_sigmoid = brier_score_loss(y_val, val_proba_sig)

    # Isotonic regression
    cal_isotonic = CalibratedClassifierCV(classifier, cv="prefit", method="isotonic")
    cal_isotonic.fit(X_val_pp, y_val)
    val_proba_iso = cal_isotonic.predict_proba(X_val_pp)[:, 1]
    brier_isotonic = brier_score_loss(y_val, val_proba_iso)

    logger.info(
        f"Calibration (Brier score, lower=better): "
        f"uncalibrated={brier_uncal:.4f}  sigmoid={brier_sigmoid:.4f}  "
        f"isotonic={brier_isotonic:.4f}"
    )

    # Select best calibration method
    brier_scores = {
        "none": (brier_uncal, classifier),
        "sigmoid": (brier_sigmoid, cal_sigmoid),
        "isotonic": (brier_isotonic, cal_isotonic),
    }
    best_cal_name, (best_brier, best_classifier) = min(brier_scores.items(), key=lambda x: x[1][0])

    # Only use calibration if it improves Brier score by > 1%
    improvement_threshold = 0.01
    if best_cal_name != "none" and (brier_uncal - best_brier) > improvement_threshold * brier_uncal:
        logger.info(
            f"Using calibration: {best_cal_name} (Brier improvement: "
            f"{brier_uncal - best_brier:.4f})"
        )
        calibration_method = best_cal_name
        final_classifier = best_classifier
    else:
        logger.info(
            "Calibration did not improve Brier score meaningfully. Using uncalibrated model."
        )
        calibration_method = "none"
        final_classifier = classifier

    # ------------------------------------------------------------------
    # 10. Final evaluation — TEST SET (evaluated exactly once)
    # ------------------------------------------------------------------
    logger.info("Evaluating on held-out test set (evaluated exactly once)...")
    test_proba = final_classifier.predict_proba(X_test_pp)[:, 1]
    test_metrics = compute_metrics(y_test.values, test_proba, selected_threshold)

    val_metrics = compute_metrics(y_val.values, val_proba, selected_threshold)

    logger.info("=" * 50)
    logger.info("VALIDATION METRICS:")
    for k, v in val_metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    logger.info("TEST METRICS (final holdout):")
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    logger.info("=" * 50)

    # ------------------------------------------------------------------
    # 11. Assemble the serializable pipeline
    # ------------------------------------------------------------------
    # We reassemble the sklearn Pipeline with the fitted components so that
    # pipeline.predict_proba(raw_df) works end-to-end at inference.
    inference_pipeline = Pipeline(
        steps=[
            ("feature_engineering", fe),
            ("preprocessor", preprocessor),
            ("classifier", final_classifier),
        ]
    )

    # ------------------------------------------------------------------
    # 12. Save pipeline
    # ------------------------------------------------------------------
    pipeline_path = config["model_assets"]["pipeline_path"]
    os.makedirs(os.path.dirname(pipeline_path), exist_ok=True)
    joblib.dump(inference_pipeline, pipeline_path)
    logger.info(f"Pipeline saved to {pipeline_path}")

    # ------------------------------------------------------------------
    # 13. Save model metadata
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    metadata = {
        "model_version": "1.1.0",
        "training_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "training_duration_seconds": round(elapsed, 1),
        "dataset": {
            "path": train_path,
            "hash_md5_first10mb": dataset_hash(train_path),
            "n_rows": len(df),
            "n_features_raw": df.shape[1],
        },
        "split_strategy": {
            "train_fraction": 0.70,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "stratified": True,
            "random_seed": seed,
        },
        "feature_engineering": {
            "version": "1.1.0",
            "bin_cols": list(fe.bin_edges_.keys()),
            "n_final_features": len(fe.final_features_),
        },
        "model": {
            "type": "LGBMClassifier",
            "hyperparameters": best_params,
            "calibration_method": calibration_method,
        },
        "threshold": {
            "selected": selected_threshold,
            "strategy": "f1_optimal_on_val",
            "all_candidates": threshold_candidates,
        },
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "environment": {
            "python_version": platform.python_version(),
            "sklearn_version": sklearn.__version__,
            "lightgbm_version": lgb.__version__,
            "platform": platform.platform(),
        },
        "git_sha": get_git_sha(),
    }

    metadata_path = os.path.join(os.path.dirname(pipeline_path), "model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Model metadata saved to {metadata_path}")

    logger.info("Training complete.")
    logger.info(
        f"  ROC-AUC (test): {test_metrics['roc_auc']:.4f}  "
        f"PR-AUC (test): {test_metrics['pr_auc']:.4f}  "
        f"Brier (test): {test_metrics['brier_score']:.4f}"
    )


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    run_training(config_path=cfg_path)
