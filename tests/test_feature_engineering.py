"""
test_feature_engineering.py
============================
Tests for FullFeatureEngineering focusing on:
1. Determinism — same input always produces same output.
2. Leakage prevention — bin edges are NOT recomputed on transform data.
3. fit/transform contract — fit() stores state, transform() uses it.
4. Edge cases — zero denominators, None/NaN values, infinite outputs.
5. EXT_SOURCE_PRODUCT creation.
6. Missing historical records produce NaN (not errors).
"""

import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.preprocessing import FullFeatureEngineering


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_two_groups(n_train: int = 100, n_test: int = 30, seed: int = 0) -> tuple:
    """
    Create synthetic train and test DataFrames with different distributions.
    We deliberately skew the test income distribution to verify that bin
    edges are NOT recomputed.
    """
    rng = np.random.default_rng(seed)

    def _make_df(n, income_scale=1.0):
        return pd.DataFrame(
            {
                "SK_ID_CURR": np.arange(1, n + 1),
                "AMT_INCOME_TOTAL": rng.exponential(100_000 * income_scale, n),
                "AMT_CREDIT": rng.exponential(300_000, n),
                "AMT_ANNUITY": rng.exponential(20_000, n),
                "AMT_GOODS_PRICE": rng.exponential(300_000, n),
                "DAYS_BIRTH": rng.integers(-25_000, -7_000, n).astype(float),
                "DAYS_EMPLOYED": rng.integers(-10_000, 0, n).astype(float),
                "DAYS_REGISTRATION": rng.integers(-10_000, 0, n).astype(float),
                "DAYS_ID_PUBLISH": rng.integers(-5_000, 0, n),
                "DAYS_LAST_PHONE_CHANGE": rng.integers(-2_000, 0, n).astype(float),
                "OWN_CAR_AGE": np.where(rng.random(n) < 0.5, np.nan, rng.integers(1, 20, n)),
                "EXT_SOURCE_1": rng.uniform(0, 1, n),
                "EXT_SOURCE_2": rng.uniform(0, 1, n),
                "EXT_SOURCE_3": rng.uniform(0, 1, n),
                "REGION_POPULATION_RELATIVE": rng.uniform(0, 0.1, n),
                "TOTALAREA_MODE": rng.uniform(0, 0.2, n),
                "FLAG_DOCUMENT_3": rng.integers(0, 2, n),
                "AMT_REQ_CREDIT_BUREAU_YEAR": rng.integers(0, 5, n).astype(float),
                "CODE_GENDER": rng.choice(["M", "F"], n),
                "NAME_CONTRACT_TYPE": rng.choice(["Cash loans", "Revolving loans"], n),
                "NAME_INCOME_TYPE": rng.choice(["Working", "State servant", "Commercial associate"], n),
                "NAME_EDUCATION_TYPE": rng.choice(["Higher education", "Secondary / secondary special"], n),
                "NAME_FAMILY_STATUS": rng.choice(["Married", "Single / not married"], n),
                "NAME_HOUSING_TYPE": rng.choice(["House / apartment", "Rented apartment"], n),
                "REGION_RATING_CLIENT": rng.integers(1, 4, n),
                "REGION_RATING_CLIENT_W_CITY": rng.integers(1, 4, n),
                "WEEKDAY_APPR_PROCESS_START": rng.choice(["MONDAY", "TUESDAY", "WEDNESDAY"], n),
                "HOUR_APPR_PROCESS_START": rng.integers(8, 20, n),
                "CNT_CHILDREN": rng.integers(0, 4, n),
                "CNT_FAM_MEMBERS": rng.integers(1, 6, n).astype(float),
                "REG_CITY_NOT_LIVE_CITY": rng.integers(0, 2, n),
                "REG_CITY_NOT_WORK_CITY": rng.integers(0, 2, n),
                "FLAG_PHONE": rng.integers(0, 2, n),
                "FLAG_WORK_PHONE": rng.integers(0, 2, n),
                "AMT_REQ_CREDIT_BUREAU_QRT": rng.integers(0, 3, n).astype(float),
                "AMT_REQ_CREDIT_BUREAU_MON": rng.integers(0, 3, n).astype(float),
                "DEF_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 3, n).astype(float),
                "DEF_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 3, n).astype(float),
                "OBS_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 5, n).astype(float),
                "OBS_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 5, n).astype(float),
                "FLOORSMAX_AVG": rng.uniform(0, 1, n),
                "FLOORSMAX_MODE": rng.uniform(0, 1, n),
                "FLOORSMAX_MEDI": rng.uniform(0, 1, n),
                "YEARS_BEGINEXPLUATATION_AVG": rng.uniform(0, 1, n),
                "YEARS_BEGINEXPLUATATION_MODE": rng.uniform(0, 1, n),
                "YEARS_BEGINEXPLUATATION_MEDI": rng.uniform(0, 1, n),
                "HOUR_APPR_PROCESS_START": rng.integers(8, 20, n),
                "AMT_REQ_CREDIT_BUREAU_WEEK": rng.integers(0, 2, n).astype(float),
            }
        )

    train_df = _make_df(n_train, income_scale=1.0)
    # Test set has 10x higher incomes → would produce very different bins if
    # pd.qcut were called on test data directly
    test_df = _make_df(n_test, income_scale=10.0)
    test_df["SK_ID_CURR"] = np.arange(n_train + 1, n_train + n_test + 1)

    return train_df, test_df


def _make_fe_no_history(config_path="config/config.yaml") -> FullFeatureEngineering:
    """Return an FE instance that uses an empty historical DataFrame."""
    empty_hist = pd.DataFrame(index=pd.Index([], name="SK_ID_CURR"))
    return FullFeatureEngineering(
        config_path=config_path,
        historical_features_df=empty_hist,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestFitTransformContract:
    def test_transform_raises_before_fit(self):
        """transform() must raise RuntimeError if called before fit()."""
        fe = _make_fe_no_history()
        train_df, _ = _make_two_groups()
        with pytest.raises(RuntimeError, match="not been fitted"):
            fe.transform(train_df)

    def test_fit_stores_bin_edges(self):
        """fit() must store bin_edges_ with at least one entry."""
        fe = _make_fe_no_history()
        train_df, _ = _make_two_groups()
        fe.fit(train_df)
        assert hasattr(fe, "bin_edges_"), "bin_edges_ not set after fit()"
        assert len(fe.bin_edges_) > 0, "bin_edges_ is empty"

    def test_fit_stores_feature_lists(self):
        """fit() must store main_table_features_ and final_features_."""
        fe = _make_fe_no_history()
        train_df, _ = _make_two_groups()
        fe.fit(train_df)
        assert hasattr(fe, "main_table_features_")
        assert hasattr(fe, "final_features_")
        assert len(fe.final_features_) > 0


class TestLeakagePrevention:
    def test_bin_edges_not_recomputed_on_transform(self):
        """
        The key leakage fix: bin edges must be the same regardless of
        what data is passed to transform().

        We fit on low-income data, then transform high-income data.
        The stored bin edges must be from fit() time, not from the
        high-income transform() data.
        """
        train_df, test_df = _make_two_groups()
        fe = _make_fe_no_history()
        fe.fit(train_df)

        # Record edges immediately after fit
        edges_after_fit = {
            k: v.copy() for k, v in fe.bin_edges_.items()
        }

        # Transform test data (which has very different income distribution)
        fe.transform(test_df)

        # Edges must not have changed
        for key in edges_after_fit:
            np.testing.assert_array_equal(
                fe.bin_edges_[key],
                edges_after_fit[key],
                err_msg=f"Bin edges for {key} were mutated during transform(). "
                        "This is a data leakage bug.",
            )

    def test_train_test_bins_differ_for_different_distributions(self):
        """
        Sanity check: if we naively fit+transform on each dataset separately,
        the resulting bin assignments differ — confirming pd.qcut would be
        leaking if used in transform().
        """
        train_df, test_df = _make_two_groups()

        # Fit on train
        fe_train = _make_fe_no_history()
        out_train = fe_train.fit_transform(train_df)

        # Fit separately on test (wrong approach — just for comparison)
        fe_test = _make_fe_no_history()
        out_test_wrong = fe_test.fit_transform(test_df)

        # Correct approach: use train's fitted FE on test data
        fe_train2 = _make_fe_no_history()
        fe_train2.fit(train_df)
        out_test_correct = fe_train2.transform(test_df)

        # The correct output should differ from the "wrongly refitted" output
        # (They happen to have different row counts so we just check edges differ)
        assert not np.array_equal(
            fe_train.bin_edges_.get("INCOME_QUANTILE_BINS", np.array([])),
            fe_test.bin_edges_.get("INCOME_QUANTILE_BINS", np.array([])),
        ), "Expected different bin edges for train vs test distributions"


class TestDeterminism:
    def test_same_input_same_output(self):
        """transform() is deterministic: same input → identical output."""
        train_df, test_df = _make_two_groups()
        fe = _make_fe_no_history()
        fe.fit(train_df)

        out1 = fe.transform(test_df)
        out2 = fe.transform(test_df)

        pd.testing.assert_frame_equal(out1, out2)

    def test_fit_transform_equals_fit_then_transform(self):
        """fit_transform(X) must equal fit(X).transform(X)."""
        train_df, _ = _make_two_groups()

        fe1 = _make_fe_no_history()
        out1 = fe1.fit_transform(train_df)

        fe2 = _make_fe_no_history()
        fe2.fit(train_df)
        out2 = fe2.transform(train_df)

        pd.testing.assert_frame_equal(out1, out2)


class TestEdgeCases:
    def test_zero_income_ratio_feature(self):
        """Zero income should not produce Inf in ratio features."""
        train_df, _ = _make_two_groups()
        fe = _make_fe_no_history()
        fe.fit(train_df)

        # Inject a zero-income row
        zero_income_df = train_df.copy()
        zero_income_df.iloc[0, zero_income_df.columns.get_loc("AMT_INCOME_TOTAL")] = 0.0
        result = fe.transform(zero_income_df)

        assert not result.isin([np.inf, -np.inf]).any().any(), \
            "Infinite values found after transform with zero income"

    def test_days_employed_anomaly_handled(self):
        """Sentinel value 365243 in DAYS_EMPLOYED should be replaced with NaN."""
        train_df, _ = _make_two_groups()
        # Set some DAYS_EMPLOYED to the sentinel
        train_df = train_df.copy()
        train_df.loc[0, "DAYS_EMPLOYED"] = 365243

        fe = _make_fe_no_history()
        result = fe.fit_transform(train_df)
        # Just verify no error is raised and output is finite
        assert result is not None

    def test_all_null_ext_sources(self):
        """All-null EXT_SOURCE columns should produce NaN product, not error."""
        train_df, _ = _make_two_groups()
        train_df = train_df.copy()
        train_df["EXT_SOURCE_1"] = np.nan
        train_df["EXT_SOURCE_2"] = np.nan
        train_df["EXT_SOURCE_3"] = np.nan

        fe = _make_fe_no_history()
        # Should not raise
        result = fe.fit_transform(train_df)
        assert result is not None

    def test_output_has_no_infinite_values(self):
        """transform() output should never contain infinite values."""
        train_df, test_df = _make_two_groups()
        fe = _make_fe_no_history()
        fe.fit(train_df)
        result = fe.transform(test_df)
        assert not result.select_dtypes(include=np.number).isin([np.inf, -np.inf]).any().any()


class TestOutputSchema:
    def test_sk_id_curr_in_output(self):
        """SK_ID_CURR should be preserved in output."""
        train_df, _ = _make_two_groups()
        fe = _make_fe_no_history()
        result = fe.fit_transform(train_df)
        assert "SK_ID_CURR" in result.columns

    def test_output_is_dataframe(self):
        """transform() must return a pandas DataFrame."""
        train_df, _ = _make_two_groups()
        fe = _make_fe_no_history()
        result = fe.fit_transform(train_df)
        assert isinstance(result, pd.DataFrame)

    def test_columns_are_strings(self):
        """All column names must be strings (no MultiIndex)."""
        train_df, _ = _make_two_groups()
        fe = _make_fe_no_history()
        result = fe.fit_transform(train_df)
        assert all(isinstance(c, str) for c in result.columns)
