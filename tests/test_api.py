import pytest
from fastapi.testclient import TestClient
import pandas as pd
import yaml
import numpy as np
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import your FastAPI app and schema from the src directory
from src.app import app
from src.api_schema import LoanApplication

# Create a TestClient instance that can be used by all tests
client = TestClient(app)

# --- Fixture to provide sample data ---
@pytest.fixture
def sample_data():
    """Fixture to load the small, saved test sample."""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    return pd.read_csv(config['data_paths']['test_sample'])

# --- Test Functions ---

def test_read_root():
    """Tests the root endpoint for a successful 'ok' response."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "API is running"}

@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_predict_endpoint_valid_input(sample_data):
    """
    Tests the /predict endpoint with a valid data sample to ensure it
    returns a successful response and a valid prediction probability.
    """
    # Use the data provided by the fixture
    sample_input = sample_data.head(1).drop(columns=['TARGET'])
    
    # Replace any NaN values with None for JSON compatibility
    sample_input.replace({np.nan: None}, inplace=True)
    
    # Convert the single-row DataFrame to a dictionary
    sample_data_json = sample_input.to_dict(orient='records')[0]

    # Send a POST request to the /predict endpoint with the sample data
    response = client.post("/predict", json=sample_data_json)
    
    # Assert that the request was successful
    assert response.status_code == 200
    
    # Assert that the response contains a valid probability
    response_json = response.json()
    assert "default_probability" in response_json
    assert isinstance(response_json["default_probability"], str)
    assert 0 <= float(response_json["default_probability"]) <= 1