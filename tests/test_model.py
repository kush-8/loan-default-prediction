import pytest
import joblib
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Load config to get paths
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

PIPELINE_PATH = config['model_assets']['pipeline_path']
SAMPLE_DATA_PATH = config['data_paths']['test_sample']

@pytest.fixture
def sample_data():
    """Fixture to load a small sample of raw data for testing."""
    return pd.read_csv(SAMPLE_DATA_PATH).sample(n=100, random_state=42)

@pytest.fixture
def trained_pipeline():
    """Fixture to load the trained model pipeline."""
    return joblib.load(PIPELINE_PATH)

@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_pipeline_prediction(trained_pipeline, sample_data):
    """Tests that the pipeline can make a prediction without errors."""
    X_sample = sample_data.drop(columns=['TARGET'])
    prediction = trained_pipeline.predict_proba(X_sample)
    assert prediction is not None

@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_prediction_output_format(trained_pipeline, sample_data):
    """Tests that the prediction output is a valid probability."""
    X_sample = sample_data.drop(columns=['TARGET'])
    prediction_proba = trained_pipeline.predict_proba(X_sample)[:, 1]
    
    assert isinstance(prediction_proba[0], float)
    assert 0 <= prediction_proba[0] <= 1

@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_model_performance(trained_pipeline, sample_data):
    """Tests that the model performance on a sample is above a minimum threshold."""
    X_sample = sample_data.drop(columns=['TARGET'])
    y_sample = sample_data['TARGET']
    
    prediction_proba = trained_pipeline.predict_proba(X_sample)[:, 1]
    auc_score = roc_auc_score(y_sample, prediction_proba)
    
    assert auc_score > 0.70