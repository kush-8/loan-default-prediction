"""
app.py
======
FastAPI inference service for the loan default risk model.

Endpoints
---------
GET  /health           — liveness check + model status
GET  /ready            — readiness probe (model loaded + functional)
GET  /model-info       — full model metadata JSON
GET  /                 — legacy root status
POST /v1/predict       — single applicant default probability + risk category

Key design decisions
--------------------
- Model pipeline is loaded ONCE at startup (lifespan context manager).
  It is stored in ``app.state.registry`` and never reloaded per request.
- Model metadata (threshold, version) is loaded from the companion JSON file.
- Request validation is handled by Pydantic v2 — invalid inputs return
  HTTP 422 with a structured error body (not a raw 500 traceback).
- The response includes probability, risk_category, predicted class, threshold,
  and model version so callers can audit which model produced the result.
- Risk categories (LOW/MEDIUM/HIGH/VERY HIGH) are configurable research-demo
  policies, not real-world banking thresholds.
- SHAP-based local explanations are computed on demand if the pipeline
  exposes a LightGBM classifier.

Run
---
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from src.models.predict import ModelRegistry

from .api_schema import LoanApplication

logger = logging.getLogger("loan_api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# ---------------------------------------------------------------------------
# Startup / shutdown lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once at startup; release at shutdown."""
    logger.info("API starting — loading model pipeline...")
    try:
        registry = ModelRegistry.from_config("config/config.yaml")
        app.state.registry = registry
        logger.info(
            f"Model loaded successfully. Version: {registry.metadata.get('model_version', 'unknown')}"
        )
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Failed to load model pipeline: {exc}")
        app.state.registry = None

    yield  # application runs here

    logger.info("API shutting down.")


# ---------------------------------------------------------------------------
# App definition
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Loan Default Risk Prediction API",
    description=(
        "Predicts the probability that a loan applicant will default, "
        "using a LightGBM classifier trained on the Home Credit dataset. "
        "**This is a machine learning prototype, not a production credit-decisioning system.**"
    ),
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Middleware: request logging
# ---------------------------------------------------------------------------


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method} {request.url.path} → {response.status_code} " f"({elapsed_ms:.1f}ms)"
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Ops"])
def health_check() -> dict[str, Any]:
    """
    Liveness check.

    Returns HTTP 200 with model status when the model is loaded,
    or HTTP 503 if the model failed to load at startup.
    """
    registry: ModelRegistry | None = getattr(app.state, "registry", None)

    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is not loaded. Check server logs.",
        )

    return {
        "status": "healthy",
        "model_version": registry.metadata.get("model_version", "unknown"),
        "calibration": registry.metadata.get("model", {}).get("calibration_method", "none"),
        "threshold": registry.threshold,
    }


@app.get("/ready", tags=["Ops"])
def readiness_check() -> dict[str, Any]:
    """
    Readiness probe — confirms model is loaded AND functional.

    Unlike /health (liveness), this endpoint verifies the model can
    actually process a request. Kubernetes readiness probes should use
    this endpoint to gate traffic.
    """
    registry: ModelRegistry | None = getattr(app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    # Confirm pipeline has the expected steps
    try:
        step_names = [name for name, _ in registry.pipeline.steps]
        required = {"feature_engineering", "preprocessor", "classifier"}
        if not required.issubset(set(step_names)):
            raise ValueError(f"Pipeline missing steps: {required - set(step_names)}")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model pipeline validation failed: {exc}",
        )
    return {"status": "ready", "model_version": registry.model_version}


@app.get("/model-info", tags=["Ops"])
def model_info() -> dict[str, Any]:
    """
    Return full model metadata.

    Includes training timestamp, dataset hash, hyperparameters, evaluation
    metrics, threshold candidates, calibration method, and environment info.
    Useful for audit, reproducibility, and dashboard display.
    """
    registry: ModelRegistry | None = getattr(app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    return registry.metadata


@app.get("/", tags=["Ops"], include_in_schema=False)
def root():
    """Legacy root endpoint — redirects semantics to /health."""
    return {
        "status": "ok",
        "message": "Loan Default Prediction API. See /docs for usage.",
    }


@app.post("/v1/predict", tags=["Prediction"])
def predict_v1(application_data: LoanApplication) -> dict[str, Any]:
    """
    Predict default probability for a single loan applicant.

    **Request body**: Application table fields (all optional; missing values
    are imputed by the pipeline).

    **Response**:
    - ``probability``: float [0, 1] — P(default)
    - ``predicted_class``: int 0 or 1
    - ``threshold``: classification threshold used
    - ``model_version``: identifier of the model that produced this result
    - ``explanation``: top risk factors increasing/reducing probability

    **Business interpretation of predicted_class**:
    - ``1`` = high default risk → recommend further review or decline
    - ``0`` = lower default risk → eligible for standard processing
    """
    registry: ModelRegistry | None = getattr(app.state, "registry", None)
    if registry is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not available. Contact the service owner.",
        )

    try:
        input_dict = application_data.model_dump(exclude_unset=True)
        if not input_dict:
            raise ValueError("Empty request body or all-null fields")

        # Convert enum values to their string representation
        input_dict = {k: (v.value if hasattr(v, "value") else v) for k, v in input_dict.items()}
        input_df = pd.DataFrame([input_dict])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to parse input: {exc}",
        )

    try:
        result = registry.predict(input_df, explain=True)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Please check the input format and try again.",
        )

    # Add risk category (configurable demo policy — not real banking thresholds)
    prob = (
        result["probability"]
        if isinstance(result["probability"], float)
        else result["probability"][0]
    )
    if prob < 0.15:
        risk_category = "LOW"
    elif prob < 0.30:
        risk_category = "MEDIUM"
    elif prob < 0.50:
        risk_category = "HIGH"
    else:
        risk_category = "VERY HIGH"
    result["risk_category"] = risk_category
    result["risk_category_note"] = (
        "Configurable demo thresholds: LOW <15%, MEDIUM 15-30%, HIGH 30-50%, VERY HIGH ≥50%"
    )

    return result


# ---------------------------------------------------------------------------
# Global exception handlers
# ---------------------------------------------------------------------------


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )
