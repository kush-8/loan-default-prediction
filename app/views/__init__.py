"""
app/views/__init__.py
=====================
Empty init so we can import from app.views in streamlit_app.py
"""

from app.views import (  # noqa: F401
    calibration,
    data,
    explainability,
    feature_engineering,
    live_prediction,
    mlops,
    model_comparison,
    monitoring,
    overview,
    performance,
)
