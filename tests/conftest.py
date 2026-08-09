"""
conftest.py
===========
Shared pytest fixtures for the loan default prediction test suite.

All fixtures that require files (config, model pipeline, sample data)
use module scope so they are loaded once per test session rather than
once per test function.

Working directory
-----------------
pytest must be run from the project root (loan-default-prediction/).
The conftest sets up sys.path so all src.* imports resolve correctly.
"""

import json
import sys
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import pytest
import yaml

# Ensure the project root is on sys.path so `from src.X import Y` works
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def config(project_root: Path) -> dict:
    """Load the project configuration once per session."""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def sample_data(config: dict) -> pd.DataFrame:
    """Load the stratified test sample (data/processed/test_sample.csv)."""
    path = config["data_paths"]["test_sample"]
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def sample_X(sample_data: pd.DataFrame) -> pd.DataFrame:
    """Features only (TARGET dropped)."""
    return sample_data.drop(columns=["TARGET"], errors="ignore")


@pytest.fixture(scope="session")
def sample_y(sample_data: pd.DataFrame) -> pd.Series:
    """Target column from test sample."""
    return sample_data["TARGET"]


# ---------------------------------------------------------------------------
# Minimal synthetic row (no file I/O required)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def minimal_valid_row() -> Dict:
    """
    A minimal application-table row with enough fields to exercise
    the feature engineering without requiring historical data files.
    """
    return {
        "SK_ID_CURR": 100001,
        "AMT_INCOME_TOTAL": 150000.0,
        "AMT_CREDIT": 450000.0,
        "AMT_ANNUITY": 22500.0,
        "AMT_GOODS_PRICE": 450000.0,
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "M",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 0,
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Single / not married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "REGION_POPULATION_RELATIVE": 0.018801,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -1800,
        "DAYS_REGISTRATION": -3000.0,
        "DAYS_ID_PUBLISH": -2000,
        "FLAG_MOBIL": 1,
        "FLAG_EMP_PHONE": 0,
        "FLAG_WORK_PHONE": 0,
        "FLAG_CONT_MOBILE": 1,
        "FLAG_PHONE": 0,
        "FLAG_EMAIL": 0,
        "OCCUPATION_TYPE": "Laborers",
        "CNT_FAM_MEMBERS": 1.0,
        "REGION_RATING_CLIENT": 2,
        "REGION_RATING_CLIENT_W_CITY": 2,
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
        "HOUR_APPR_PROCESS_START": 10,
        "REG_REGION_NOT_LIVE_REGION": 0,
        "REG_REGION_NOT_WORK_REGION": 0,
        "LIVE_REGION_NOT_WORK_REGION": 0,
        "REG_CITY_NOT_LIVE_CITY": 0,
        "REG_CITY_NOT_WORK_CITY": 0,
        "LIVE_CITY_NOT_WORK_CITY": 0,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.4,
        "OWN_CAR_AGE": None,
        "TOTALAREA_MODE": 0.05,
        "DAYS_LAST_PHONE_CHANGE": -300.0,
        "FLAG_DOCUMENT_3": 1,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
        "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
        "AMT_REQ_CREDIT_BUREAU_MON": 0.0,
        "AMT_REQ_CREDIT_BUREAU_WEEK": 0.0,
    }


@pytest.fixture(scope="session")
def minimal_valid_df(minimal_valid_row: Dict) -> pd.DataFrame:
    """Single-row DataFrame from the minimal valid row dict."""
    return pd.DataFrame([minimal_valid_row])


# ---------------------------------------------------------------------------
# Model pipeline (optional — skipped if not found)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def trained_pipeline(config: dict):
    """
    Load the saved sklearn pipeline.
    Tests that require this fixture are skipped if the file doesn't exist.
    """
    import joblib
    path = config["model_assets"]["pipeline_path"]
    if not os.path.exists(path):
        pytest.skip(f"Trained pipeline not found at {path}. Run train.py first.")
    return joblib.load(path)


@pytest.fixture(scope="session")
def model_metadata(config: dict) -> dict:
    """Load model metadata JSON (optional)."""
    import os
    pipeline_dir = os.path.dirname(config["model_assets"]["pipeline_path"])
    meta_path = os.path.join(pipeline_dir, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            return json.load(f)
    return {}
