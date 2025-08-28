import pandas as pd
import joblib
import shap
import yaml
import matplotlib.pyplot as plt
import os

def generate_explanations():
    """
    Loads the final pipeline and generates and saves SHAP explainability plots.
    """
    print("--- Generating model explanations ---")
    
    # 1. Load Assets
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    pipeline = joblib.load(config['model_assets']['pipeline_path'])
    df = pd.read_csv(config['data_paths']['application_train'])
    
    X_sample = df.sample(n=100, random_state=42).drop(columns=['TARGET'])
    
    # 2. Extract All Pipeline Steps
    feature_engineering = pipeline.named_steps['feature_engineering']
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['classifier']
    
    # 3. Apply the Full Preprocessing Pipeline
    # Step A: Apply custom feature engineering
    X_sample_engineered = feature_engineering.transform(X_sample)
    
    # Step B: Apply standard preprocessing (scaling, encoding, etc.)
    X_sample_processed = preprocessor.transform(X_sample_engineered)
    
    # Reconstruct a DataFrame for SHAP
    feature_names = preprocessor.get_feature_names_out()
    X_sample_processed_df = pd.DataFrame(X_sample_processed, columns=feature_names)

    # 4. Create SHAP Explainer and Values
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample_processed_df)
    
    # 5. Generate and Save Global and Local Plots
    # (The rest of the script is the same)
    shap.summary_plot(shap_values, X_sample_processed_df, show=False)
    plt.savefig('reports/shap_summary_plot.png', bbox_inches='tight')
    plt.close()
    print("Global summary plot saved to reports/shap_summary_plot.png")

    force_plot_html = shap.force_plot(
        shap_values.base_values[0],
        shap_values.values[0],
        X_sample_processed_df.iloc[0],
        matplotlib=False
    )
    shap.save_html('reports/shap_force_plot_single.html', force_plot_html)
    print("Local force plot saved to reports/shap_force_plot_single.html")

if __name__ == '__main__':
    os.makedirs('reports', exist_ok=True)
    generate_explanations()