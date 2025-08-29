import pandas as pd
import pytest
import yaml

# Load config to get data paths
with open('config/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

SAMPLE_DATA_PATH = config['data_paths']['test_sample']

@pytest.fixture
def sample_data():
    """A pytest fixture to load the raw data once for all tests."""
    return pd.read_csv(SAMPLE_DATA_PATH)

def test_data_shape(sample_data):
    """Tests if the raw data has a reasonable number of rows and columns."""
    assert sample_data.shape[0] > 1000
    assert sample_data.shape[1] > 100

def test_required_columns_exist(sample_data):
    """Tests for the presence of essential columns identified during EDA."""
    required_cols = [
        'SK_ID_CURR', 'TARGET', 'AMT_INCOME_TOTAL', 'AMT_CREDIT',
        'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'CODE_GENDER', 'DAYS_BIRTH',
        'DAYS_EMPLOYED', 'NAME_EDUCATION_TYPE', 'EXT_SOURCE_1',
        'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]
    assert all(col in sample_data.columns for col in required_cols)

def test_target_column_is_binary(sample_data):
    """Tests if the TARGET column contains only 0s and 1s."""
    assert set(sample_data['TARGET'].unique()) == {0, 1}

def test_id_column_is_unique(sample_data):
    """Tests that the primary key column is unique."""
    assert sample_data['SK_ID_CURR'].is_unique

def test_numerical_columns_have_correct_type(sample_data):
    """Tests if key numerical columns are of a numeric dtype."""
    numeric_cols_to_check = [
        'AMT_CREDIT', 'AMT_INCOME_TOTAL', 'AMT_ANNUITY',
        'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'DAYS_EMPLOYED',
        'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'
    ]
    assert all(pd.api.types.is_numeric_dtype(sample_data[col]) for col in numeric_cols_to_check)