import pytest
import joblib
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
import sys
import os

# Add the src directory to the Python path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src import preprocessing

# --- Fixtures to load assets once per test session ---
@pytest.fixture(scope="module")
def config():
    """Fixture to load the project configuration."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture(scope="module")
def sample_data(config):
    """Fixture to load the small, saved test sample data."""
    return pd.read_csv(config['data_paths']['test_sample'])

@pytest.fixture(scope="module")
def trained_pipeline(config):
    """Fixture to load the trained model pipeline."""
    return joblib.load(config['model_assets']['pipeline_path'])

# --- Mock Function ---
def mock_feature_engineering(df, config=None):
    """A dummy function that bypasses the real feature engineering during tests."""
    return df

# --- Test Functions ---

@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_pipeline_prediction(trained_pipeline, sample_data, monkeypatch):
    """Tests that the pipeline can make a prediction without errors."""
    # Temporarily replace the real functions that need large data files
    monkeypatch.setattr(preprocessing, 'feature_engineer_bureau_data', mock_feature_engineering)
    monkeypatch.setattr(preprocessing, 'feature_engineer_previous_data', mock_feature_engineering)

    X_sample = sample_data.drop(columns=['TARGET'])
    prediction = trained_pipeline.predict_proba(X_sample)
    assert prediction is not None

@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_prediction_output_format(trained_pipeline, sample_data, monkeypatch):
    """Tests that the prediction output is a valid probability."""
    monkeypatch.setattr(preprocessing, 'feature_engineer_bureau_data', mock_feature_engineering)
    monkeypatch.setattr(preprocessing, 'feature_engineer_previous_data', mock_feature_engineering)

    X_sample = sample_data.drop(columns=['TARGET'])
    prediction_proba = trained_pipeline.predict_proba(X_sample)[:, 1]
    
    assert isinstance(prediction_proba[0], float)
    assert 0 <= prediction_proba[0] <= 1

@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_model_performance(trained_pipeline, sample_data, monkeypatch):
    """Tests that the model performance on a sample is above a minimum threshold."""
    monkeypatch.setattr(preprocessing, 'feature_engineer_bureau_data', mock_feature_engineering)
    monkeypatch.setattr(preprocessing, 'feature_engineer_previous_data', mock_feature_engineering)

    X_sample = sample_data.drop(columns=['TARGET'])
    y_sample = sample_data['TARGET']
    
    prediction_proba = trained_pipeline.predict_proba(X_sample)[:, 1]
    auc_score = roc_auc_score(y_sample, prediction_proba)
    
    # Note: The performance will be lower than the real model because we are not using the
    # powerful historical features. We just check that it's above a basic sanity threshold.
    assert auc_score > 0.70