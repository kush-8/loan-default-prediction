# tests/test_ci.py
from fastapi.testclient import TestClient
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.app import app

def test_api_health_check():
    """
    A simple test to ensure the API server starts and the root endpoint works.
    This test is self-contained and has no external dependencies.
    """
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"