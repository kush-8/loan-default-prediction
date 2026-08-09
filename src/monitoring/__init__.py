"""
src/monitoring/__init__.py
==========================
Monitoring utilities for the loan default prediction model.

Provides functions for:
- Feature drift detection (PSI, KS-statistic, Wasserstein distance)
- Prediction drift monitoring
- Calibration drift assessment
- Demo data generation for dashboard visualization

All functions that return simulated/demonstration data are clearly
documented as such. Real monitoring requires a production data store.
"""

from src.monitoring.drift import (
    compute_feature_drift_report,
    compute_ks_statistic,
    compute_psi,
    compute_wasserstein,
)

__all__ = [
    "compute_psi",
    "compute_ks_statistic",
    "compute_wasserstein",
    "compute_feature_drift_report",
]
