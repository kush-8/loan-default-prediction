import pytest
import joblib
import pandas as pd
import yaml

@pytest.mark.filterwarnings("ignore:X does not have valid feature names")
def test_full_pipeline_integration():
    """
    Tests the entire pipeline from raw data to prediction.
    """
    # 1. Load Config and a sample of raw data
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    df = pd.read_csv(config['data_paths']['application_train'])
    sample_data = df.sample(n=50, random_state=42)
    
    X_sample = sample_data.drop(columns=['TARGET'])

    # 2. Load the full, trained pipeline
    pipeline = joblib.load(config['model_assets']['pipeline_path'])

    # 3. Make predictions
    predictions = pipeline.predict_proba(X_sample)[:, 1]

    # 4. Assertions
    # Check that we get a prediction for every row in the sample
    assert len(predictions) == len(X_sample)
    
    # Check that all predictions are valid probabilities (between 0 and 1)
    assert all(0 <= pred <= 1 for pred in predictions)