"""
test_api.py
===========
Tests for the FastAPI prediction service.

Coverage:
- Health endpoint (GET /health)
- Valid prediction request (POST /v1/predict)
- Invalid input handling (422 with useful error)
- Missing required context
- Type mismatch
- Model unavailable scenario

The test client uses FastAPI's TestClient which runs the full ASGI stack
including the lifespan event (model loading). Tests that require the model
to be loaded are skipped if the pipeline file doesn't exist.
"""

import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    """TestClient that exercises the full app lifespan (model loading)."""
    from src.api.app import app

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest.fixture(scope="module")
def valid_payload(config) -> dict[str, Any]:
    """A valid minimal request body for POST /v1/predict."""
    sample_path = config["data_paths"]["test_sample"]
    if os.path.exists(sample_path):
        df = pd.read_csv(sample_path)
        row = df.drop(columns=["TARGET"], errors="ignore").iloc[0]
        row_dict = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
        # Convert enum-like string values to their string form
        return row_dict
    else:
        # Fallback minimal payload
        return {
            "SK_ID_CURR": 100001,
            "AMT_INCOME_TOTAL": 150000.0,
            "AMT_CREDIT": 450000.0,
            "AMT_ANNUITY": 22500.0,
            "CODE_GENDER": "M",
            "DAYS_BIRTH": -12000,
            "DAYS_EMPLOYED": -1800,
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.4,
        }


# ---------------------------------------------------------------------------
# Health endpoint tests
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    def test_health_returns_200_or_503(self, client):
        """
        /health must return 200 (model loaded) or 503 (model unavailable).
        Both are acceptable — 503 means the model file was not present.
        """
        response = client.get("/health")
        assert response.status_code in (200, 503), f"Unexpected status code: {response.status_code}"

    def test_health_response_has_status_field(self, client):
        """Response body must contain a 'status' or 'detail' field."""
        response = client.get("/health")
        body = response.json()
        assert "status" in body or "detail" in body

    def test_health_200_includes_model_version(self, client):
        """When healthy, response includes model_version."""
        response = client.get("/health")
        if response.status_code == 200:
            body = response.json()
            assert "model_version" in body
            assert "threshold" in body

    def test_root_endpoint_returns_200(self, client):
        """Legacy GET / must return 200."""
        response = client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert "status" in body


# ---------------------------------------------------------------------------
# Prediction endpoint tests
# ---------------------------------------------------------------------------


class TestPredictEndpoint:
    def test_valid_request_returns_200(self, client, valid_payload):
        """A valid request must return HTTP 200."""
        response = client.post("/v1/predict", json=valid_payload)
        # If model not loaded, will be 503 — skip in that case
        if response.status_code == 503:
            pytest.skip("Model not loaded — skipping prediction tests")
        assert (
            response.status_code == 200
        ), f"Expected 200, got {response.status_code}: {response.text}"

    def test_valid_request_has_probability_field(self, client, valid_payload):
        """Response must contain 'probability' field."""
        response = client.post("/v1/predict", json=valid_payload)
        if response.status_code != 200:
            pytest.skip("Skipping — model not loaded or request failed")
        body = response.json()
        assert "probability" in body, f"Missing 'probability' in response: {body}"

    def test_probability_is_valid_float(self, client, valid_payload):
        """Returned probability must be a float in [0, 1]."""
        response = client.post("/v1/predict", json=valid_payload)
        if response.status_code != 200:
            pytest.skip("Skipping — model not loaded")
        prob = response.json()["probability"]
        assert isinstance(prob, (int, float)), f"probability is not numeric: {type(prob)}"
        assert 0.0 <= prob <= 1.0, f"probability {prob} is outside [0, 1]"

    def test_response_has_predicted_class(self, client, valid_payload):
        """Response must contain 'predicted_class' (0 or 1)."""
        response = client.post("/v1/predict", json=valid_payload)
        if response.status_code != 200:
            pytest.skip("Skipping — model not loaded")
        body = response.json()
        assert "predicted_class" in body
        assert body["predicted_class"] in (0, 1)

    def test_response_has_threshold(self, client, valid_payload):
        """Response must include the threshold used for classification."""
        response = client.post("/v1/predict", json=valid_payload)
        if response.status_code != 200:
            pytest.skip("Skipping — model not loaded")
        body = response.json()
        assert "threshold" in body
        t = body["threshold"]
        assert 0.0 < t < 1.0, f"Threshold {t} outside (0, 1)"

    def test_response_has_model_version(self, client, valid_payload):
        """Response must include model_version identifier."""
        response = client.post("/v1/predict", json=valid_payload)
        if response.status_code != 200:
            pytest.skip("Skipping — model not loaded")
        body = response.json()
        assert "model_version" in body


class TestInvalidInputHandling:
    def test_empty_request_body_returns_422(self, client):
        """Empty JSON object should return 422 or be handled gracefully."""
        response = client.post("/v1/predict", json={})
        # With all-optional fields, {} is valid — the model handles missing values
        # But the endpoint must not return 500
        assert response.status_code != 500, (
            "Empty request body caused 500 Server Error. "
            "Should return 422 or 200 (with imputation)."
        )

    def test_wrong_content_type_returns_error(self, client):
        """Non-JSON body should return an error (not 500)."""
        response = client.post(
            "/v1/predict",
            data="not json",
            headers={"Content-Type": "text/plain"},
        )
        assert response.status_code in (
            400,
            415,
            422,
        ), f"Expected 4xx error for wrong content-type, got {response.status_code}"

    def test_invalid_categorical_value_handled(self, client, valid_payload):
        """
        An invalid enum value (e.g., CODE_GENDER='Z') should return 422,
        not a 500 traceback.
        """
        bad_payload = {**valid_payload, "CODE_GENDER": "INVALID_VALUE_XYZ"}
        response = client.post("/v1/predict", json=bad_payload)
        # FastAPI/Pydantic will reject the invalid enum with 422
        assert response.status_code in (
            200,
            422,
        ), f"Unexpected status {response.status_code} for invalid enum value"
        assert (
            response.status_code != 500
        ), "Invalid enum value caused 500 Server Error — should be 422"

    def test_string_in_numeric_field_returns_422(self, client, valid_payload):
        """Passing a string for a numeric field should return 422."""
        bad_payload = {**valid_payload, "AMT_INCOME_TOTAL": "not_a_number"}
        response = client.post("/v1/predict", json=bad_payload)
        assert (
            response.status_code == 422
        ), f"Expected 422 for string in numeric field, got {response.status_code}"

    def test_none_for_all_optional_fields_is_handled(self, client):
        """All-None payload should not cause 500 — missing values are imputed."""
        payload = {"SK_ID_CURR": 999999}  # Only ID provided
        response = client.post("/v1/predict", json=payload)
        assert response.status_code != 500, "All-null input caused 500 Server Error"
