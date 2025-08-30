import pytest
import joblib
import pandas as pd
import yaml
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src import preprocessing

# --- Fixtures to load assets ---
@pytest.fixture(scope="module")
def config():
    """Fixture to load the project configuration."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def sample_data(config):
    """Fixture to load the entire test sample data."""
    return pd.read_csv(config['data_paths']['test_sample'])

@pytest.fixture(scope="module")
def trained_pipeline(config):
    """Fixture to load the trained model pipeline."""
    return joblib.load(config['model_assets']['pipeline_path'])

# --- Mock Function ---
def mock_feature_engineering(df, config=None):
    """A dummy function to replace the real feature engineering during tests."""
    return df

# --- Integration Test ---
@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_full_pipeline_integration(trained_pipeline, sample_data, monkeypatch):
    """
    Tests the entire pipeline from raw sample data to prediction, mocking the
    historical data dependencies for a self-contained CI test.
    """
    # 1. Mock the functions that depend on large, missing data files
    monkeypatch.setattr(preprocessing, 'feature_engineer_bureau_data', mock_feature_engineering)
    monkeypatch.setattr(preprocessing, 'feature_engineer_previous_data', mock_feature_engineering)

    # 2. Prepare the sample data
    X_sample = sample_data.drop(columns=['TARGET'], errors='ignore')

    # 3. Make predictions using the full pipeline
    predictions = trained_pipeline.predict_proba(X_sample)[:, 1]

    # 4. Assertions
    # Check that we get a prediction for every row in the sample
    assert len(predictions) == len(X_sample)
    
    # Check that all predictions are valid probabilities (between 0 and 1)
    assert all(0 <= pred <= 1 for pred in predictions)