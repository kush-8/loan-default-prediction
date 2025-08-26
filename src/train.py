import pandas as pd
import yaml
import json
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Importing custom preprocessing classes and functions
from preprocessing import FullFeatureEngineering, create_preprocessor

def run_training():
    """
    Trains and saves the final model pipeline.
    """
    print("--- Starting training pipeline ---")

    # 1. Load Config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 2. Load Raw Data
    df = pd.read_csv(config['data_paths']['application_train'])

    # 3. Train-Test Split
    X = df.drop(columns=[config['training_params']['target_column']])
    y = df[config['training_params']['target_column']]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config['training_params']['test_size'],
        random_state=config['training_params']['random_state'],
        stratify=y
    )

    # 4. Define the Full Model Pipeline
    # First, get the column types after feature engineering
    temp_df = FullFeatureEngineering().fit_transform(X_train)
    numerical_cols = temp_df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = temp_df.select_dtypes(include='object').columns.tolist()
    if 'SK_ID_CURR' in numerical_cols:
        numerical_cols.remove('SK_ID_CURR')

    # Load best hyperparameters
    with open(config['model_assets']['model_parameters'], 'r') as f:
        best_params = json.load(f)

    # Create the full pipeline
    final_pipeline = Pipeline(steps=[
        ('feature_engineering', FullFeatureEngineering()),
        ('preprocessor', create_preprocessor(numerical_cols, categorical_cols)),
        ('classifier', lgb.LGBMClassifier(**best_params))
    ])

    # 5. Train the Pipeline
    print("Training the final end-to-end pipeline...")
    final_pipeline.fit(X_train, y_train)

    # 6. Evaluate the Pipeline
    y_pred_proba = final_pipeline.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nFinal Pipeline Test ROC AUC Score: {auc:.4f}")

    # 7. Save the Final Pipeline
    joblib.dump(final_pipeline, config['model_assets']['pipeline_path'])
    print(f"Final pipeline saved to {config['model_assets']['pipeline_path']}")

if __name__ == '__main__':
    run_training()