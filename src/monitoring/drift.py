"""
drift.py
========
Statistical drift detection utilities for the loan default prediction model.

Provides production-ready, reusable functions for detecting:
- Feature distribution drift (Population Stability Index, KS-statistic, Wasserstein)
- Prediction score drift
- These functions are model-agnostic and can be used in any monitoring pipeline.

Design principles
-----------------
- All functions are pure (no global state, no side effects).
- All functions accept numpy arrays or pandas Series.
- Missing values are handled gracefully (dropped before computation).
- Functions return typed dicts suitable for JSON serialization.

Usage
-----
    from src.monitoring.drift import compute_psi, compute_feature_drift_report

    psi = compute_psi(reference_scores, current_scores, n_bins=10)
    report = compute_feature_drift_report(reference_df, current_df, features)
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Population Stability Index (PSI)
# ---------------------------------------------------------------------------

# PSI interpretation thresholds (widely used in credit risk monitoring)
_PSI_STABLE = 0.1  # < 0.10: No significant shift
_PSI_MODERATE = 0.25  # 0.10 – 0.25: Moderate shift, investigate
# > 0.25: Significant shift, model likely needs retraining


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    """
    Compute the Population Stability Index (PSI) between two distributions.

    PSI measures how much a distribution has shifted relative to a reference.
    Originally developed for credit scorecards; widely used in model monitoring.

    Interpretation
    --------------
    PSI < 0.10  : Stable — no significant shift
    PSI 0.10–0.25 : Moderate shift — investigate
    PSI > 0.25  : Significant shift — consider retraining

    Parameters
    ----------
    reference : array-like
        Reference distribution (e.g., training data probabilities).
    current : array-like
        Current distribution (e.g., recent production probabilities).
    n_bins : int
        Number of bins for discretisation.
    epsilon : float
        Small value to avoid log(0) or division by zero.

    Returns
    -------
    dict with keys: psi, n_bins, status, reference_n, current_n
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        logger.warning("PSI: empty array after dropping NaN — returning NaN")
        return {"psi": float("nan"), "n_bins": n_bins, "status": "insufficient_data"}

    # Define bin edges from the reference distribution
    _, bin_edges = np.histogram(ref, bins=n_bins)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=bin_edges)
    cur_counts, _ = np.histogram(cur, bins=bin_edges)

    ref_pct = (ref_counts + epsilon) / (len(ref) + epsilon * n_bins)
    cur_pct = (cur_counts + epsilon) / (len(cur) + epsilon * n_bins)

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))

    if psi < _PSI_STABLE:
        status = "stable"
    elif psi < _PSI_MODERATE:
        status = "moderate_shift"
    else:
        status = "significant_shift"

    return {
        "psi": round(psi, 6),
        "n_bins": n_bins,
        "status": status,
        "reference_n": int(len(ref)),
        "current_n": int(len(cur)),
    }


# ---------------------------------------------------------------------------
# Kolmogorov-Smirnov Test
# ---------------------------------------------------------------------------


def compute_ks_statistic(
    reference: np.ndarray,
    current: np.ndarray,
) -> dict[str, Any]:
    """
    Compute the two-sample Kolmogorov-Smirnov (KS) statistic.

    The KS statistic is the maximum absolute difference between the
    empirical CDFs of the two distributions. The p-value tests the null
    hypothesis that both samples were drawn from the same distribution.

    Parameters
    ----------
    reference : array-like
    current : array-like

    Returns
    -------
    dict with keys: ks_statistic, p_value, significant (p < 0.05)
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) < 2 or len(cur) < 2:
        return {"ks_statistic": float("nan"), "p_value": float("nan"), "significant": False}

    result = stats.ks_2samp(ref, cur)
    return {
        "ks_statistic": round(float(result.statistic), 6),
        "p_value": round(float(result.pvalue), 6),
        "significant": bool(result.pvalue < 0.05),
    }


# ---------------------------------------------------------------------------
# Wasserstein Distance (Earth Mover's Distance)
# ---------------------------------------------------------------------------


def compute_wasserstein(
    reference: np.ndarray,
    current: np.ndarray,
) -> dict[str, float]:
    """
    Compute the Wasserstein-1 distance (Earth Mover's Distance).

    Unlike PSI, Wasserstein distance is not sensitive to binning choices
    and has a natural geometric interpretation: the minimum cost to
    transform one distribution into the other.

    Parameters
    ----------
    reference : array-like
    current : array-like

    Returns
    -------
    dict with keys: wasserstein_distance
    """
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    if len(ref) == 0 or len(cur) == 0:
        return {"wasserstein_distance": float("nan")}

    dist = float(stats.wasserstein_distance(ref, cur))
    return {"wasserstein_distance": round(dist, 6)}


# ---------------------------------------------------------------------------
# Full feature drift report
# ---------------------------------------------------------------------------


def compute_feature_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list[str],
    n_bins: int = 10,
) -> dict[str, dict[str, Any]]:
    """
    Compute a drift report for a list of features across two DataFrames.

    For each feature, computes PSI, KS-statistic, and Wasserstein distance.
    Only numerical features are supported; categorical features are skipped
    with a warning.

    Parameters
    ----------
    reference_df : pd.DataFrame
        Reference distribution data (e.g., training data).
    current_df : pd.DataFrame
        Current production data.
    features : list[str]
        Feature names to evaluate. Must be present in both DataFrames.
    n_bins : int
        Bins for PSI computation.

    Returns
    -------
    dict mapping feature_name -> {psi, ks_statistic, p_value, wasserstein_distance, status}
    """
    report = {}

    for feat in features:
        if feat not in reference_df.columns or feat not in current_df.columns:
            logger.warning(f"Feature '{feat}' not found in both DataFrames — skipping.")
            continue

        if not pd.api.types.is_numeric_dtype(reference_df[feat]):
            logger.warning(f"Feature '{feat}' is not numeric — skipping drift computation.")
            report[feat] = {"status": "skipped_non_numeric"}
            continue

        ref_vals = reference_df[feat].dropna().values
        cur_vals = current_df[feat].dropna().values

        psi_result = compute_psi(ref_vals, cur_vals, n_bins=n_bins)
        ks_result = compute_ks_statistic(ref_vals, cur_vals)
        ws_result = compute_wasserstein(ref_vals, cur_vals)

        report[feat] = {
            "psi": psi_result["psi"],
            "psi_status": psi_result["status"],
            "ks_statistic": ks_result["ks_statistic"],
            "ks_p_value": ks_result["p_value"],
            "ks_significant": ks_result["significant"],
            "wasserstein_distance": ws_result["wasserstein_distance"],
            "reference_n": int(len(ref_vals)),
            "current_n": int(len(cur_vals)),
        }

    logger.info(
        f"Drift report complete: {len(report)} features evaluated, "
        f"{sum(1 for v in report.values() if v.get('psi_status') == 'significant_shift')} "
        f"significant shifts detected."
    )
    return report


# ---------------------------------------------------------------------------
# Prediction score drift
# ---------------------------------------------------------------------------


def compute_prediction_drift(
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
) -> dict[str, Any]:
    """
    Compute drift in model prediction scores (probabilities).

    Returns PSI, KS test, Wasserstein distance, and summary statistics
    for both reference and current score distributions.

    Parameters
    ----------
    reference_scores : array-like
        Predicted probabilities from the reference period.
    current_scores : array-like
        Predicted probabilities from the current period.

    Returns
    -------
    dict with drift metrics and distribution statistics.
    """
    ref = np.asarray(reference_scores, dtype=float)
    cur = np.asarray(current_scores, dtype=float)
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]

    def _stats(arr: np.ndarray) -> dict:
        if len(arr) == 0:
            return {}
        return {
            "mean": round(float(arr.mean()), 6),
            "std": round(float(arr.std()), 6),
            "p10": round(float(np.percentile(arr, 10)), 6),
            "p50": round(float(np.percentile(arr, 50)), 6),
            "p90": round(float(np.percentile(arr, 90)), 6),
            "predicted_positive_rate": round(float((arr >= 0.5).mean()), 6),
            "n": int(len(arr)),
        }

    return {
        "psi": compute_psi(ref, cur)["psi"],
        "psi_status": compute_psi(ref, cur)["status"],
        "ks_statistic": compute_ks_statistic(ref, cur)["ks_statistic"],
        "ks_p_value": compute_ks_statistic(ref, cur)["p_value"],
        "wasserstein_distance": compute_wasserstein(ref, cur)["wasserstein_distance"],
        "reference_stats": _stats(ref),
        "current_stats": _stats(cur),
    }
