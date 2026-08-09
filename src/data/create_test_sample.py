# scripts/create_test_sample.py
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config['data_paths']['application_train'])

_, sample_df = train_test_split(
    df, 
    test_size=0.01, 
    random_state=42, 
    stratify=df['TARGET']
)

sample_df.to_csv(config['data_paths']['test_sample'], index=False)
print(f"Test sample saved to {config['data_paths']['test_sample']}")