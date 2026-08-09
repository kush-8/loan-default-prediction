"""
test_model.py
=============
Tests for model loading, prediction shape, probability range,
threshold logic, and regression (golden output).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestModelLoading:
    def test_pipeline_is_sklearn_pipeline(self, trained_pipeline):
        """Loaded artifact must be a sklearn Pipeline."""
        from sklearn.pipeline import Pipeline

        assert isinstance(trained_pipeline, Pipeline)

    def test_pipeline_has_required_steps(self, trained_pipeline):
        """Pipeline must have feature_engineering, preprocessor, classifier."""
        step_names = [name for name, _ in trained_pipeline.steps]
        assert "feature_engineering" in step_names, "Missing feature_engineering step"
        assert "preprocessor" in step_names, "Missing preprocessor step"
        assert "classifier" in step_names, "Missing classifier step"

    def test_feature_engineering_is_fitted(self, trained_pipeline):
        """FullFeatureEngineering step must have been fitted (has bin_edges_)."""
        fe = trained_pipeline.named_steps["feature_engineering"]
        assert hasattr(fe, "bin_edges_"), (
            "FullFeatureEngineering is not fitted — bin_edges_ missing. "
            "This means the saved pipeline was trained with the old (leaky) code."
        )


class TestPredictionShape:
    def test_predict_proba_shape(self, trained_pipeline, sample_X):
        """predict_proba must return shape (n_samples, 2)."""
        proba = trained_pipeline.predict_proba(sample_X)
        assert proba.ndim == 2
        assert proba.shape[0] == len(sample_X)
        assert proba.shape[1] == 2

    def test_predict_proba_single_row(self, trained_pipeline, sample_X):
        """Single-row prediction must return shape (1, 2)."""
        single = sample_X.iloc[:1]
        proba = trained_pipeline.predict_proba(single)
        assert proba.shape == (1, 2)

    def test_predict_binary_output(self, trained_pipeline, sample_X):
        """predict() must return binary 0/1 values."""
        preds = trained_pipeline.predict(sample_X)
        assert set(np.unique(preds)).issubset({0, 1})


class TestProbabilityConstraints:
    def test_probabilities_between_0_and_1(self, trained_pipeline, sample_X):
        """All predicted probabilities must be in [0, 1]."""
        proba = trained_pipeline.predict_proba(sample_X)[:, 1]
        assert (proba >= 0).all(), f"Found {(proba < 0).sum()} negative probabilities"
        assert (proba <= 1).all(), f"Found {(proba > 1).sum()} probabilities > 1"

    def test_probabilities_sum_to_one(self, trained_pipeline, sample_X):
        """Class probabilities must sum to 1 per row."""
        proba = trained_pipeline.predict_proba(sample_X)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, 1.0, atol=1e-6, err_msg="Class probabilities do not sum to 1"
        )

    def test_no_nan_in_predictions(self, trained_pipeline, sample_X):
        """Predictions must not contain NaN."""
        proba = trained_pipeline.predict_proba(sample_X)[:, 1]
        assert not np.isnan(proba).any(), f"Found {np.isnan(proba).sum()} NaN predictions"

    def test_no_all_same_predictions(self, trained_pipeline, sample_X):
        """The model must produce varying probabilities (not all the same)."""
        proba = trained_pipeline.predict_proba(sample_X)[:, 1]
        assert proba.std() > 0.01, (
            f"All predictions are nearly identical (std={proba.std():.6f}). "
            "The model may not be working correctly."
        )


class TestPerformance:
    def test_roc_auc_above_baseline(self, trained_pipeline, sample_X, sample_y):
        """ROC-AUC must be above the random baseline (0.5) with margin."""
        from sklearn.metrics import roc_auc_score

        proba = trained_pipeline.predict_proba(sample_X)[:, 1]
        auc = roc_auc_score(sample_y, proba)
        assert auc > 0.65, (
            f"ROC-AUC={auc:.4f} is too low. "
            "A trained model should significantly outperform random."
        )

    def test_roc_auc_upper_bound(self, trained_pipeline, sample_X, sample_y):
        """ROC-AUC must not be suspiciously high (would indicate leakage)."""
        from sklearn.metrics import roc_auc_score

        proba = trained_pipeline.predict_proba(sample_X)[:, 1]
        auc = roc_auc_score(sample_y, proba)
        assert auc < 0.99, f"ROC-AUC={auc:.4f} is suspiciously high. Check for data leakage."


class TestThresholdLogic:
    def test_metadata_has_threshold(self, model_metadata):
        """Model metadata must contain a threshold value."""
        if not model_metadata:
            pytest.skip("model_metadata.json not found")
        assert "threshold" in model_metadata, "No 'threshold' key in metadata"
        t = model_metadata["threshold"].get("selected")
        assert t is not None, "No 'selected' threshold in metadata"
        assert 0.0 < t < 1.0, f"Threshold {t} is outside (0, 1)"

    def test_threshold_gives_sensible_recall(
        self, trained_pipeline, sample_X, sample_y, model_metadata
    ):
        """At the selected threshold, recall should be reasonable (>= 0.3)."""
        if not model_metadata:
            pytest.skip("model_metadata.json not found")
        from sklearn.metrics import recall_score

        threshold = model_metadata.get("threshold", {}).get("selected", 0.5)
        proba = trained_pipeline.predict_proba(sample_X)[:, 1]
        preds = (proba >= threshold).astype(int)
        recall = recall_score(sample_y, preds, zero_division=0)
        assert recall >= 0.30, (
            f"Recall={recall:.3f} at threshold={threshold:.3f} is too low. "
            "Model may be predicting almost all negative."
        )


class TestDeterminism:
    def test_same_input_same_prediction(self, trained_pipeline, sample_X):
        """Predictions must be identical across multiple calls."""
        proba1 = trained_pipeline.predict_proba(sample_X)[:, 1]
        proba2 = trained_pipeline.predict_proba(sample_X)[:, 1]
        np.testing.assert_array_equal(
            proba1,
            proba2,
            err_msg="Predictions differ across calls — model is not deterministic",
        )
