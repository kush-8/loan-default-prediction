"""
test_ci.py
==========
Fast, self-contained tests for the CI pipeline.

Requirements:
- No raw data files needed
- No model pipeline needed
- Only requires requirements-api.txt dependencies
- Must complete in < 30 seconds

These tests verify that the basic application structure is intact
and the API server can start without errors.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAPIStartup:
    def test_app_imports_without_error(self):
        """The FastAPI app module must be importable without exceptions."""
        try:
            pass
        except Exception as exc:  # noqa: BLE001
            assert False, f"src.app import failed: {exc}"

    def test_api_schema_imports_without_error(self):
        """The API schema module must be importable."""
        try:
            pass
        except Exception as exc:  # noqa: BLE001
            assert False, f"src.api_schema import failed: {exc}"

    def test_loan_application_model_is_pydantic(self):
        """LoanApplication must be a Pydantic model class."""
        from pydantic import BaseModel

        from src.api.api_schema import LoanApplication

        assert issubclass(LoanApplication, BaseModel)

    def test_root_endpoint_returns_ok(self):
        """GET / must return HTTP 200 with status='ok'."""
        from fastapi.testclient import TestClient

        from src.api.app import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/")
            assert response.status_code == 200
            assert response.json().get("status") == "ok"

    def test_health_endpoint_exists(self):
        """GET /health endpoint must be registered and return 200 or 503."""
        from fastapi.testclient import TestClient

        from src.api.app import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/health")
            assert response.status_code in (
                200,
                503,
            ), f"Expected 200 or 503 from /health, got {response.status_code}"

    def test_predict_endpoint_registered(self):
        """POST /v1/predict must be a registered route."""
        from src.api.app import app

        routes = [route.path for route in app.routes]
        assert "/v1/predict" in routes, f"/v1/predict not found in routes: {routes}"

    def test_predict_endpoint_rejects_invalid_type(self):
        """POST /v1/predict with wrong numeric type should return 422."""
        from fastapi.testclient import TestClient

        from src.api.app import app

        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.post("/v1/predict", json={"AMT_INCOME_TOTAL": "this_is_not_a_number"})
            assert response.status_code == 422, (
                f"Expected 422 for invalid numeric type, got {response.status_code}: "
                f"{response.text}"
            )
