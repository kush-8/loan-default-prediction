import json
from enum import Enum
from typing import Any

import yaml
from pydantic import Field, create_model


def create_loan_application_schema(config_path: str):
    """
    Creates the Pydantic schema from saved JSON files, including dynamic Enums.
    """
    # Load all necessary configuration and schema files
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    with open(config["data_paths"]["api_schema"], "r") as f:
        schema = json.load(f)

    with open(config["data_paths"]["column_descriptions"], "r") as f:
        description_dict = json.load(f)

    with open(config["data_paths"]["categorical_enums"], "r") as f:
        enums_dict = json.load(f)

    # Define how to map pandas dtypes to Python types for the API
    type_mapping = {"int64": int | None, "float64": float | None}

    fields = {}
    # Loop through every column defined in our data schema
    for col, dtype in schema.items():
        description = description_dict.get(col, "No description available.")

        # Check if this column is categorical by seeing if it's in our enums file
        if col in enums_dict:
            # Dynamically create an Enum with the pre-saved unique values
            unique_values = {str(val): str(val) for val in enums_dict.get(col, [])}
            DynamicEnum = Enum(f"{col}Enum", unique_values)
            fields[col] = (DynamicEnum | None, Field(None, description=description))
        else:
            # Handle numerical types
            python_type = type_mapping.get(dtype, Any)
            fields[col] = (python_type, Field(None, description=description))

    return create_model("LoanApplication", **fields)


# Create the schema once when the module is loaded
LoanApplication = create_loan_application_schema("config/config.yaml")
