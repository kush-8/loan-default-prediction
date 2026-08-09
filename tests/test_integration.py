"""
test_integration.py
===================
Integration tests for the full pipeline (raw input → preprocessed → prediction).

These tests use the loaded pipeline artifact (requires models/final_pipeline.joblib)
but do NOT need raw historical CSV files because the historical features are
pre-joined in the serialized pipeline's feature engineering step.

Unlike the old tests, we do NOT monkeypatch away the historical feature loading —
the new pipeline uses pre-computed historical features, which means:
  - If historical_features.parquet exists: full pipeline runs correctly.
  - If it doesn't: FullFeatureEngineering.transform() will raise FileNotFoundError,
    which is the correct behavior (historical cache must be pre-built).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFullPipelineIntegration:
    def test_pipeline_runs_on_sample(self, trained_pipeline, sample_X):
        """Full pipeline (FE → preprocess → predict) runs on test sample without error."""
        predictions = trained_pipeline.predict_proba(sample_X)[:, 1]
        assert len(predictions) == len(sample_X)

    def test_predictions_are_valid_probabilities(self, trained_pipeline, sample_X):
        """All predictions from the full pipeline are in [0, 1]."""
        predictions = trained_pipeline.predict_proba(sample_X)[:, 1]
        assert (predictions >= 0).all()
        assert (predictions <= 1).all()

    def test_no_nans_in_output(self, trained_pipeline, sample_X):
        """No NaN values in predictions."""
        predictions = trained_pipeline.predict_proba(sample_X)[:, 1]
        assert not np.isnan(predictions).any()

    def test_pipeline_output_length_matches_input(self, trained_pipeline, sample_X):
        """Output length must match input length."""
        predictions = trained_pipeline.predict_proba(sample_X)
        assert predictions.shape[0] == len(sample_X)
        assert predictions.shape[1] == 2

    def test_single_row_inference(self, trained_pipeline, sample_X):
        """Single-row prediction must work correctly."""
        single_row = sample_X.iloc[:1].copy()
        pred = trained_pipeline.predict_proba(single_row)[:, 1]
        assert len(pred) == 1
        assert 0.0 <= pred[0] <= 1.0

    def test_inference_with_missing_values(self, trained_pipeline, sample_X):
        """
        Pipeline must handle additional missing values gracefully.
        The preprocessing pipeline has a SimpleImputer that handles NaN.
        """
        X_with_nulls = sample_X.copy()
        # Introduce extra NaN values in some columns
        for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "AMT_GOODS_PRICE"]:
            if col in X_with_nulls.columns:
                X_with_nulls.loc[X_with_nulls.index[:5], col] = np.nan

        predictions = trained_pipeline.predict_proba(X_with_nulls)[:, 1]
        assert len(predictions) == len(X_with_nulls)
        assert not np.isnan(predictions).any()


class TestInferenceLatency:
    @pytest.mark.slow
    def test_single_prediction_latency(self, trained_pipeline, sample_X):
        """
        Single prediction (including full pipeline) should complete in < 5 seconds.
        Historical features are pre-joined in the pipeline, so no CSV I/O at inference.
        """
        import time

        single = sample_X.iloc[:1]
        start = time.perf_counter()
        trained_pipeline.predict_proba(single)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, (
            f"Single prediction took {elapsed:.2f}s. "
            "If historical features are being loaded from CSV on each call, "
            "the historical_features.parquet cache may not be configured correctly."
        )

    @pytest.mark.slow
    def test_batch_prediction_latency(self, trained_pipeline, sample_X):
        """Batch of 100 rows should complete in < 10 seconds."""
        import time

        batch = sample_X.iloc[:100]
        start = time.perf_counter()
        trained_pipeline.predict_proba(batch)
        elapsed = time.perf_counter() - start
        assert elapsed < 10.0, f"Batch prediction took {elapsed:.2f}s"
