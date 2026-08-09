"""
test_data.py
============
Tests for data validation: schema, types, constraints, edge cases.

These tests verify that the test_sample.csv used by the CI pipeline
meets the expected schema — catching data corruption or schema drift.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CODE_GENDER",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "NAME_EDUCATION_TYPE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

NUMERIC_COLUMNS = [
    "AMT_CREDIT",
    "AMT_INCOME_TOTAL",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

VALID_GENDER_VALUES = {"M", "F", "XNA"}
VALID_CONTRACT_TYPES = {"Cash loans", "Revolving loans"}


class TestDataSchema:
    def test_data_has_minimum_rows(self, sample_data):
        """Dataset must have at least 1000 rows to be statistically useful."""
        assert sample_data.shape[0] >= 1000, (
            f"Only {sample_data.shape[0]} rows found. " "Need at least 1000 for meaningful tests."
        )

    def test_data_has_minimum_columns(self, sample_data):
        """Dataset should have over 100 columns (raw + engineered)."""
        assert sample_data.shape[1] > 100, f"Only {sample_data.shape[1]} columns. Expected > 100."

    def test_required_columns_exist(self, sample_data):
        """All business-critical columns must be present."""
        missing = [c for c in REQUIRED_COLUMNS if c not in sample_data.columns]
        assert not missing, f"Missing required columns: {missing}"

    def test_target_column_is_binary(self, sample_data):
        """TARGET must contain exactly {0, 1}."""
        actual = set(sample_data["TARGET"].dropna().unique())
        assert actual == {0, 1}, f"Unexpected TARGET values: {actual}"

    def test_target_has_no_nulls(self, sample_data):
        """TARGET must have zero null values."""
        null_count = sample_data["TARGET"].isnull().sum()
        assert null_count == 0, f"TARGET has {null_count} null values"

    def test_id_column_is_unique(self, sample_data):
        """SK_ID_CURR must be unique (primary key)."""
        dupe_count = sample_data["SK_ID_CURR"].duplicated().sum()
        assert dupe_count == 0, f"Found {dupe_count} duplicate SK_ID_CURR values"

    def test_id_column_has_no_nulls(self, sample_data):
        """SK_ID_CURR must never be null."""
        assert sample_data["SK_ID_CURR"].isnull().sum() == 0

    def test_numeric_columns_are_numeric(self, sample_data):
        """Key numeric columns must have numeric dtype."""
        non_numeric = [
            c
            for c in NUMERIC_COLUMNS
            if c in sample_data.columns and not pd.api.types.is_numeric_dtype(sample_data[c])
        ]
        assert not non_numeric, f"Non-numeric dtype for: {non_numeric}"

    def test_gender_values_are_valid(self, sample_data):
        """CODE_GENDER must contain only known values."""
        if "CODE_GENDER" not in sample_data.columns:
            pytest.skip("CODE_GENDER not in sample")
        observed = set(sample_data["CODE_GENDER"].dropna().unique())
        unexpected = observed - VALID_GENDER_VALUES
        assert not unexpected, f"Unexpected CODE_GENDER values: {unexpected}"


class TestDataConstraints:
    def test_credit_is_positive(self, sample_data):
        """AMT_CREDIT must be positive where not null."""
        if "AMT_CREDIT" not in sample_data.columns:
            pytest.skip("AMT_CREDIT not in sample")
        non_positive = (sample_data["AMT_CREDIT"].dropna() <= 0).sum()
        assert non_positive == 0, f"Found {non_positive} non-positive AMT_CREDIT values"

    def test_income_is_positive(self, sample_data):
        """AMT_INCOME_TOTAL must be positive where not null."""
        if "AMT_INCOME_TOTAL" not in sample_data.columns:
            pytest.skip("AMT_INCOME_TOTAL not in sample")
        non_positive = (sample_data["AMT_INCOME_TOTAL"].dropna() <= 0).sum()
        assert non_positive == 0, f"Found {non_positive} non-positive AMT_INCOME_TOTAL"

    def test_days_birth_is_negative(self, sample_data):
        """DAYS_BIRTH is stored as negative (days before application)."""
        if "DAYS_BIRTH" not in sample_data.columns:
            pytest.skip("DAYS_BIRTH not in sample")
        positive_count = (sample_data["DAYS_BIRTH"].dropna() >= 0).sum()
        assert positive_count == 0, (
            f"Found {positive_count} non-negative DAYS_BIRTH values. "
            "DAYS_BIRTH should be negative."
        )

    def test_ext_sources_in_range(self, sample_data):
        """EXT_SOURCE_1/2/3 should be in [0, 1] (scores)."""
        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
            if col not in sample_data.columns:
                continue
            vals = sample_data[col].dropna()
            out_of_range = ((vals < 0) | (vals > 1)).sum()
            assert out_of_range == 0, f"{col} has {out_of_range} values outside [0, 1]"

    def test_target_class_imbalance_documented(self, sample_data):
        """
        Document that the dataset is imbalanced.
        Default rate should be between 5% and 35%.
        This is a documentation test — it fails if data is corrupted.
        """
        default_rate = sample_data["TARGET"].mean()
        assert 0.05 <= default_rate <= 0.35, (
            f"Unexpected default rate: {default_rate:.3f}. "
            "Expected 5-35% for Home Credit dataset. Data may be corrupted."
        )


class TestDataEdgeCases:
    def test_no_all_null_rows(self, sample_data):
        """Rows with all values null are invalid."""
        feature_cols = [c for c in sample_data.columns if c not in ["SK_ID_CURR", "TARGET"]]
        all_null_rows = sample_data[feature_cols].isnull().all(axis=1).sum()
        assert all_null_rows == 0, f"Found {all_null_rows} rows with all-null features"

    def test_no_infinite_values(self, sample_data):
        """Raw data should not contain infinite values."""
        numeric_data = sample_data.select_dtypes(include=np.number)
        inf_count = np.isinf(numeric_data.values).sum()
        assert inf_count == 0, f"Found {inf_count} infinite values in raw data"
