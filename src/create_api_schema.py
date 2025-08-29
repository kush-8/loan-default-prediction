import pandas as pd
import json
import yaml

# Load configuration
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# --- 1. Generate data schema from application_train.csv ---
df_train = pd.read_csv(config['data_paths']['application_train'])
df_train = df_train.drop(columns=['TARGET'])
schema = {col: str(dtype) for col, dtype in df_train.dtypes.items()}

with open(config['data_paths']['api_schema'], 'w') as f:
    json.dump(schema, f, indent=4)
print(f"API data schema saved to {config['data_paths']['api_schema']}")

# --- 2. Generate descriptions schema from descriptions CSV ---
descriptions_df = pd.read_csv(config['data_paths']['column_descriptions_csv'], encoding='ISO-8859-1')
app_descriptions = descriptions_df[descriptions_df['Table'] == 'application_{train|test}.csv']
description_dict = app_descriptions.set_index('Row')['Description'].to_dict()

with open(config['data_paths']['column_descriptions'], 'w') as f:
    json.dump(description_dict, f, indent=4)
print(f"Column descriptions saved to {config['data_paths']['column_descriptions']}")