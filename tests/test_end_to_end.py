import pytest
import pandas as pd
import joblib
import os
import yaml
import time
from sklearn.pipeline import Pipeline

# Load config to get paths
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

@pytest.mark.e2e
def test_final_artifacts_exist():
    """
    Tests if the key artifacts were created by the training pipeline.
    """
        
    assert os.path.exists(config['model_assets']['pipeline_path'])
    assert os.path.exists('results.csv')
    assert os.path.exists('reports/shap_summary_plot.png')

@pytest.mark.e2e
def test_result_file_format():
    """
    Tests if the result file has the correct format and valid predictions.
    """
    result_df = pd.read_csv('results.csv')
    
    assert 'SK_ID_CURR' in result_df.columns
    assert 'TARGET' in result_df.columns
    assert result_df['TARGET'].isnull().sum() == 0
    assert all(0 <= pred <= 1 for pred in result_df['TARGET'])

@pytest.mark.e2e
def test_model_pipeline_loads_and_predicts():
    """Tests loading the final pipeline and making a prediction on the test sample."""
    pipeline = joblib.load(config['model_assets']['pipeline_path'])
    sample_df = pd.read_csv(config['data_paths']['test_sample'])
    
    # Drop target if it exists in the sample
    if 'TARGET' in sample_df.columns:
        X_sample = sample_df.drop(columns=['TARGET'])
    else:
        X_sample = sample_df
        
    predictions = pipeline.predict(X_sample)
    assert len(predictions) == len(X_sample)

@pytest.mark.e2e
def test_prediction_latency():
    """
    Tests that the model's inference latency on preprocessed data
    is within an acceptable limit.
    """
    pipeline = joblib.load(config['model_assets']['pipeline_path'])
    sample_df = pd.read_csv(config['data_paths']['test_sample'])
    X_sample = sample_df.drop(columns=['TARGET'], errors='ignore')

    # 1. Create a preprocessing-only pipeline
    preprocessing_pipeline = Pipeline(steps=[
        ('feature_engineering', pipeline.named_steps['feature_engineering']),
        ('preprocessor', pipeline.named_steps['preprocessor'])
    ])
    
    # 2. Preprocess the data
    X_sample_processed = preprocessing_pipeline.transform(X_sample)
    
    # 3. Time only the final classifier's prediction step
    model = pipeline.named_steps['classifier']
    start_time = time.time()
    model.predict(X_sample_processed)
    end_time = time.time()
    
    latency = end_time - start_time
    print(f"\nModel-only inference latency: {latency:.4f} seconds")
    
    # The model itself should be very fast on a preprocessed sample
    assert latency < 1.0