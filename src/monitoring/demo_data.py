"""
demo_data.py
============
Generate clearly labelled demonstration data for the monitoring dashboard.

IMPORTANT
---------
ALL functions in this module generate SIMULATED data for demonstration purposes.
They do NOT represent real production traffic or real drift events.
The Streamlit dashboard displays an explicit "DEMONSTRATION DATA" banner
wherever this data is shown.

Design notes
------------
- Data is generated deterministically using a fixed random seed so the
  dashboard looks consistent across restarts.
- Data is designed to tell a realistic story: stable for several weeks,
  then a simulated drift event, followed by a recovery.
- All returned objects are plain Python dicts/lists (JSON-serializable).
"""

import numpy as np

_RNG_SEED = 42


def generate_score_drift_timeseries(
    n_weeks: int = 20,
    baseline_mean: float = 0.18,
    drift_week: int = 14,
    drift_magnitude: float = 0.07,
) -> list[dict]:
    """
    Generate weekly prediction score drift metrics for dashboard demo.

    Simulates:
    - Weeks 1–13: stable production (scores close to training distribution)
    - Weeks 14–17: simulated feature/population drift event
    - Weeks 18–20: model recalibrated / distribution normalises

    Returns
    -------
    list of dicts with: week, mean_score, psi, ks_statistic, alert
    """
    rng = np.random.default_rng(_RNG_SEED)
    weeks = []

    for w in range(1, n_weeks + 1):
        if w < drift_week:
            # Stable regime
            mean_score = baseline_mean + rng.normal(0, 0.005)
            psi = rng.uniform(0.01, 0.06)
            ks = rng.uniform(0.02, 0.06)
            alert = None
        elif w <= drift_week + 3:
            # Drift event
            progress = (w - drift_week) / 4
            mean_score = baseline_mean + drift_magnitude * progress + rng.normal(0, 0.008)
            psi = rng.uniform(0.12, 0.35)
            ks = rng.uniform(0.10, 0.30)
            alert = "drift_detected" if psi > 0.10 else None
        else:
            # Recovery
            recovery = (w - drift_week - 4) / 3
            mean_score = baseline_mean + drift_magnitude * (1 - recovery) + rng.normal(0, 0.004)
            psi = rng.uniform(0.04, 0.12)
            ks = rng.uniform(0.03, 0.09)
            alert = "investigating" if psi > 0.10 else "recovering"

        # Derive simulated weekly predicted positive rate
        pred_positive_rate = max(0, min(1, mean_score * 2.5 + rng.normal(0, 0.01)))

        weeks.append(
            {
                "week": w,
                "week_label": f"Week {w}",
                "mean_score": round(float(mean_score), 4),
                "predicted_positive_rate": round(float(pred_positive_rate), 4),
                "psi": round(float(psi), 4),
                "ks_statistic": round(float(ks), 4),
                "alert": alert,
            }
        )

    return weeks


def generate_feature_drift_report(features: list[str] | None = None) -> dict[str, dict]:
    """
    Generate a simulated per-feature drift report for dashboard demo.

    DEMONSTRATION DATA — does not reflect real production traffic.

    Returns
    -------
    dict mapping feature_name -> {psi, psi_status, ks_statistic, wasserstein_distance}
    """
    rng = np.random.default_rng(_RNG_SEED + 1)

    if features is None:
        features = [
            "EXT_SOURCE_2",
            "EXT_SOURCE_3",
            "CREDIT_INCOME_PERCENT",
            "AMT_CREDIT",
            "AMT_INCOME_TOTAL",
            "DAYS_BIRTH",
            "INSTAL_PAYMENT_PERC_MEAN",
            "BUREAU_AMT_CREDIT_SUM_DEBT_MEAN",
            "PAYMENT_RATE",
            "DAYS_EMPLOYED",
        ]

    # Assign drift severity profile: most stable, 2-3 drifted
    n = len(features)
    psi_values = np.concatenate(
        [
            rng.uniform(0.01, 0.08, n - 3),  # stable
            rng.uniform(0.10, 0.22, 2),  # moderate
            rng.uniform(0.27, 0.45, 1),  # significant
        ]
    )
    rng.shuffle(psi_values)

    report = {}
    for feat, psi in zip(features, psi_values):
        psi = float(psi)
        ks = float(np.clip(psi * rng.uniform(0.5, 1.5), 0, 1))
        ws = float(np.abs(rng.normal(0, psi * 2)))

        if psi < 0.10:
            status = "stable"
        elif psi < 0.25:
            status = "moderate_shift"
        else:
            status = "significant_shift"

        report[feat] = {
            "psi": round(psi, 4),
            "psi_status": status,
            "ks_statistic": round(ks, 4),
            "wasserstein_distance": round(ws, 4),
        }

    return report


def generate_performance_timeseries(n_weeks: int = 20, drift_week: int = 14) -> list[dict]:
    """
    Generate weekly model performance metrics for dashboard demo.

    DEMONSTRATION DATA — does not reflect real production performance.

    Returns
    -------
    list of dicts with: week, roc_auc, pr_auc, f1, precision, recall, brier
    """
    rng = np.random.default_rng(_RNG_SEED + 2)
    weeks = []

    for w in range(1, n_weeks + 1):
        if w < drift_week:
            roc_auc = 0.772 + rng.normal(0, 0.003)
            pr_auc = 0.256 + rng.normal(0, 0.005)
            f1 = 0.328 + rng.normal(0, 0.004)
            brier = 0.067 + rng.normal(0, 0.001)
        elif w <= drift_week + 3:
            decay = (w - drift_week) * 0.02
            roc_auc = 0.772 - decay + rng.normal(0, 0.004)
            pr_auc = 0.256 - decay * 1.5 + rng.normal(0, 0.006)
            f1 = 0.328 - decay * 1.2 + rng.normal(0, 0.005)
            brier = 0.067 + decay * 0.5 + rng.normal(0, 0.002)
        else:
            recovery = (w - drift_week - 4) * 0.015
            roc_auc = 0.740 + recovery + rng.normal(0, 0.003)
            pr_auc = 0.216 + recovery * 1.5 + rng.normal(0, 0.005)
            f1 = 0.292 + recovery + rng.normal(0, 0.004)
            brier = 0.073 - recovery * 0.3 + rng.normal(0, 0.002)

        precision = float(np.clip(f1 * rng.uniform(0.8, 1.2), 0, 1))
        recall = float(np.clip(f1 / max(precision, 1e-6) * f1, 0, 1))

        weeks.append(
            {
                "week": w,
                "week_label": f"Week {w}",
                "roc_auc": round(float(np.clip(roc_auc, 0, 1)), 4),
                "pr_auc": round(float(np.clip(pr_auc, 0, 1)), 4),
                "f1": round(float(np.clip(f1, 0, 1)), 4),
                "precision": round(float(np.clip(precision, 0, 1)), 4),
                "recall": round(float(np.clip(recall, 0, 1)), 4),
                "brier_score": round(float(np.clip(brier, 0, 1)), 4),
                "drift_event": w >= drift_week and w <= drift_week + 3,
            }
        )

    return weeks


def generate_prediction_volume_timeseries(
    n_weeks: int = 20,
    base_volume: int = 5000,
) -> list[dict]:
    """
    Generate weekly prediction volume for dashboard demo.

    DEMONSTRATION DATA.
    """
    rng = np.random.default_rng(_RNG_SEED + 3)
    result = []
    for w in range(1, n_weeks + 1):
        vol = int(base_volume + rng.normal(0, base_volume * 0.05))
        result.append({"week": w, "week_label": f"Week {w}", "volume": vol})
    return result


def get_demo_monitoring_summary(drift_week: int = 14) -> dict:
    """
    Get a high-level summary of the demo monitoring scenario.

    Returns
    -------
    dict with: scenario_description, drift_event_details, recommendation
    """
    return {
        "data_source": "DEMONSTRATION DATA — simulated, not real production traffic",
        "n_weeks": 20,
        "scenario": (
            f"Weeks 1–{drift_week - 1}: Stable production period. "
            f"Weeks {drift_week}–{drift_week + 3}: Simulated feature distribution shift "
            f"(e.g., economic shock changes AMT_INCOME_TOTAL distribution). "
            f"Weeks {drift_week + 4}–20: Recovery after model investigation."
        ),
        "alert_week": drift_week,
        "triggered_by": "PSI > 0.10 on EXT_SOURCE_2 and CREDIT_INCOME_PERCENT",
        "recommended_action": (
            "1. Investigate data pipeline for upstream schema changes. "
            "2. Collect labeled samples from the drift period. "
            "3. Retrain model on updated distribution if drift persists > 2 weeks."
        ),
    }
