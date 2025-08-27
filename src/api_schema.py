# src/schemas.py
import pandas as pd
from pydantic import create_model, Field
from typing import Any, Optional
from enum import Enum

def create_loan_application_schema(csv_path: str, descriptions_path: str):
    """
    Creates the Pydantic schema for the LoanApplication.
    """
    df = pd.read_csv(csv_path)
    descriptions_df = pd.read_csv(descriptions_path, encoding='ISO-8859-1')
    app_descriptions = descriptions_df[descriptions_df['Table'] == 'application_{train|test}.csv']
    description_dict = app_descriptions.set_index('Row')['Description'].to_dict()

    type_mapping = {'int64': Optional[int], 'float64': Optional[float]}

    fields = {}
    for col, dtype in df.dtypes.items():
        if col != 'TARGET':
            description = description_dict.get(col, "No description available.")
            
            if str(dtype) == 'object':
                unique_values = {val: val for val in df[col].unique() if pd.notna(val)}
                DynamicEnum = Enum(f'{col}Enum', unique_values)
                fields[col] = (Optional[DynamicEnum], Field(None, description=description))
            else:
                python_type = type_mapping.get(str(dtype), Any)
                fields[col] = (python_type, Field(None, description=description))
            
    return create_model('LoanApplication', **fields)

# Create the schema once when the module is loaded
LoanApplication = create_loan_application_schema(
    'data/raw/application_train.csv', 
    'data/raw/HomeCredit_columns_description.csv'
)