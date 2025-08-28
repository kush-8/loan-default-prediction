import pandas as pd
import pytest
import yaml

# Load config to get data paths
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

RAW_DATA_PATH = config['data_paths']['application_train']

@pytest.fixture
def raw_data():
    """A pytest fixture to load the raw data once for all tests."""
    return pd.read_csv(RAW_DATA_PATH)

def test_data_shape(raw_data):
    """Tests if the raw data has a reasonable number of rows and columns."""
    assert raw_data.shape[0] > 50000
    assert raw_data.shape[1] > 100

def test_required_columns_exist(raw_data):
    """Tests for the presence of essential columns identified during EDA."""
    required_cols = [
        'SK_ID_CURR', 'TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
        'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'CODE_GENDER', 'DAYS_BIRTH',
        'DAYS_EMPLOYED', 'NAME_EDUCATION_TYPE', 'EXT_SOURCE_1',
        'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]
    assert all(col in raw_data.columns for col in required_cols)

def test_target_column_is_binary(raw_data):
    """Tests if the TARGET column contains only 0s and 1s."""
    assert set(raw_data['TARGET'].unique()) == {0, 1}

def test_id_column_is_unique(raw_data):
    """Tests that the primary key column is unique."""
    assert raw_data['SK_ID_CURR'].is_unique

def test_numerical_columns_have_correct_type(raw_data):
    """Tests if key numerical columns are of a numeric dtype."""
    numeric_cols_to_check = [
        'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY',
        'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]
    assert all(pd.api.types.is_numeric_dtype(raw_data[col]) for col in numeric_cols_to_check)