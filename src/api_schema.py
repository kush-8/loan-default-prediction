import pandas as pd
import yaml
import json
from pydantic import create_model, Field
from typing import Any, Optional
from enum import Enum

def create_loan_application_schema(config_path: str):
    """
    Creates the Pydantic schema from saved JSON schema and description files,
    with paths defined in the main config file.
    """
    # Load the main configuration file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load the data schema (column names and types)
    with open(config['data_paths']['api_schema'], 'r') as f:
        schema = json.load(f)
        
    # Load the column descriptions
    with open(config['data_paths']['column_descriptions'], 'r') as f:
        description_dict = json.load(f)

    # Define how to map pandas dtypes to Python types for the API
    type_mapping = {'int64': Optional[int], 'float64': Optional[float]}

    fields = {}
    for col, dtype in schema.items():
        description = description_dict.get(col, "No description available.")
        
        # If the column is categorical, create an Enum for dropdowns in the API docs
        if dtype == 'object':
            # Note: For a real-time API, you might load these unique values from a saved file as well
            # For simplicity here, we still read the raw file for enum values, but this could be optimized
            raw_df = pd.read_csv(config['data_paths']['application_train'])
            unique_values = {str(val): str(val) for val in raw_df[col].unique() if pd.notna(val)}
            DynamicEnum = Enum(f'{col}Enum', unique_values)
            fields[col] = (Optional[DynamicEnum], Field(None, description=description))
        else:
            # Handle numerical types
            python_type = type_mapping.get(dtype, Any)
            fields[col] = (python_type, Field(None, description=description))
            
    return create_model('LoanApplication', **fields)

# Create the schema once when the module is loaded, using the config file
LoanApplication = create_loan_application_schema('config/config.yaml')