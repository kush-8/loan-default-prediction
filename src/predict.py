import pandas as pd
import joblib
import yaml

def load_pipeline():
    """Loads the saved prediction pipeline."""
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    pipeline = joblib.load(config['model_assets']['pipeline_path'])
    return pipeline

def make_single_prediction(input_data):
    """
    Makes a prediction for a single data point.
    
    Args:
        input_data (pd.DataFrame): A DataFrame with a single row.
    
    Returns:
        float: The predicted default probability.
    """

    # Load the full pipeline (preprocessor + model)
    pipeline = load_pipeline()

    # Pipeline handle all the preprocessing internally
    predicted_proba = pipeline.predict_proba(input_data)[:, 1]

    return predicted_proba[0]

def make_batch_predictions(input_data):
    """
    Makes predictions for a batch of data.
    
    Args:
        input_data (pd.DataFrame): A DataFrame with multiple rows.
    
    Returns:
        np.array: An array of prediction probabilities.
    """


    # Load the full pipeline (preprocessor + model)
    pipeline = load_pipeline()

    # Pipeline handle all the preprocessing internally
    predictions = pipeline.predict_proba(input_data)[:, 1]

    return predictions

if __name__ == '__main__':
    # This block allows testing the script directly
    
    # Load raw data for testing 
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    
    # --- Test Single Prediction ---
    test_df_for_sample = pd.read_csv(config['data_paths']['application_test'])
    sample_input = test_df_for_sample.head(1)
    
    single_pred = make_single_prediction(sample_input)
    print("--- Single Prediction Test ---")
    print(f"Prediction for SK_ID_CURR {sample_input['SK_ID_CURR'].iloc[0]}: {single_pred:.4f}\n")

    # --- Test Batch Prediction ---
    test_df = pd.read_csv(config['data_paths']['application_test'])
    predictions = make_batch_predictions(test_df)
    
    # Create a result file
    result_df = pd.DataFrame({
        'SK_ID_CURR': test_df['SK_ID_CURR'],
        'TARGET': predictions
    })
    
    result_df.to_csv(config['data_paths']['test_result'], index=False)

    print("--- Batch Prediction on application_test.csv Complete ---")
    print("Submission file 'result.csv' created successfully.")
    print("First 5 predictions:")
    print(result_df.head())