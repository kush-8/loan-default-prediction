"""
test_end_to_end.py
==================
End-to-end tests that verify the complete training → artifact → inference chain.

These tests require:
- A fully trained pipeline (models/final_pipeline.joblib)
- The companion metadata file (models/model_metadata.json)
- The test sample (data/processed/test_sample.csv)

Run with: pytest -m e2e
"""

import os
import sys
import time
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.mark.e2e
class TestArtifactsExist:
    def test_pipeline_artifact_exists(self, config):
        """The trained pipeline file must exist."""
        assert os.path.exists(config["model_assets"]["pipeline_path"]), (
            f"Pipeline not found at {config['model_assets']['pipeline_path']}. "
            "Run train.py first."
        )

    def test_metadata_exists(self, config):
        """model_metadata.json must exist alongside the pipeline."""
        pipeline_dir = os.path.dirname(config["model_assets"]["pipeline_path"])
        meta_path = os.path.join(pipeline_dir, "model_metadata.json")
        assert os.path.exists(meta_path), (
            f"model_metadata.json not found at {meta_path}. "
            "This file is required for version tracking and threshold loading."
        )

    def test_shap_summary_plot_exists(self):
        """SHAP summary plot should exist after explain.py is run."""
        assert os.path.exists(
            "reports/shap_summary_plot.png"
        ), "SHAP summary plot not found. Run: python src/explain.py"


@pytest.mark.e2e
class TestMetadataSchema:
    def test_metadata_has_required_fields(self, model_metadata):
        """Model metadata must have all required tracking fields."""
        if not model_metadata:
            pytest.skip("model_metadata.json not found")

        required = [
            "model_version",
            "training_timestamp",
            "threshold",
            "validation_metrics",
            "test_metrics",
            "feature_engineering",
        ]
        missing = [f for f in required if f not in model_metadata]
        assert not missing, f"Missing metadata fields: {missing}"

    def test_threshold_in_valid_range(self, model_metadata):
        """Selected threshold must be between 0 and 1."""
        if not model_metadata:
            pytest.skip("model_metadata.json not found")
        t = model_metadata.get("threshold", {}).get("selected", None)
        assert t is not None
        assert 0 < t < 1, f"Threshold {t} is not in (0, 1)"

    def test_test_metrics_have_auc(self, model_metadata):
        """Test metrics must include ROC-AUC."""
        if not model_metadata:
            pytest.skip("model_metadata.json not found")
        test_metrics = model_metadata.get("test_metrics", {})
        assert "roc_auc" in test_metrics
        auc = test_metrics["roc_auc"]
        assert 0.5 < auc < 1.0, f"ROC-AUC {auc} is outside expected range"

    def test_no_test_set_used_for_threshold(self, model_metadata):
        """Threshold strategy must reference val set, not test set."""
        if not model_metadata:
            pytest.skip("model_metadata.json not found")
        strategy = model_metadata.get("threshold", {}).get("strategy", "")
        assert (
            "test" not in strategy.lower()
        ), f"Threshold strategy '{strategy}' references test set — this is data leakage!"


@pytest.mark.e2e
class TestResultFile:
    def test_result_file_format(self):
        """results.csv must have correct columns and valid probability values."""
        if not os.path.exists("results.csv"):
            pytest.skip("results.csv not found. Run predict.py first.")
        result_df = pd.read_csv("results.csv")
        assert "SK_ID_CURR" in result_df.columns
        assert "TARGET" in result_df.columns
        assert result_df["TARGET"].isnull().sum() == 0
        assert (result_df["TARGET"] >= 0).all()
        assert (result_df["TARGET"] <= 1).all()


@pytest.mark.e2e
class TestPipelineLoadAndPredict:
    def test_pipeline_loads_and_predicts(self, trained_pipeline, sample_data):
        """Full pipeline loads correctly and produces predictions on test sample."""
        X_sample = sample_data.drop(columns=["TARGET"], errors="ignore")
        predictions = trained_pipeline.predict(X_sample)
        assert len(predictions) == len(X_sample)

    def test_pipeline_roc_auc_on_sample(self, trained_pipeline, sample_X, sample_y):
        """Pipeline achieves acceptable ROC-AUC on the held-out test sample."""
        from sklearn.metrics import roc_auc_score

        proba = trained_pipeline.predict_proba(sample_X)[:, 1]
        auc = roc_auc_score(sample_y, proba)
        assert auc > 0.70, (
            f"ROC-AUC={auc:.4f} on test sample is below acceptable threshold. "
            "Model performance may have regressed."
        )

    @pytest.mark.slow
    def test_model_only_latency(self, trained_pipeline, sample_X):
        """Classifier-only inference (post-preprocessing) should be < 1 second for the sample."""
        fe = trained_pipeline.named_steps["feature_engineering"]
        preprocessor = trained_pipeline.named_steps["preprocessor"]
        classifier = trained_pipeline.named_steps["classifier"]

        X_eng = fe.transform(sample_X)
        drop_cols = [c for c in ["SK_ID_CURR"] if c in X_eng.columns]
        X_pp = preprocessor.transform(X_eng.drop(columns=drop_cols, errors="ignore"))

        start = time.perf_counter()
        classifier.predict(X_pp)
        elapsed = time.perf_counter() - start

        print(f"\nClassifier-only latency: {elapsed:.4f}s on {len(sample_X)} rows")
        assert elapsed < 1.0, f"Classifier took {elapsed:.2f}s — unexpectedly slow"
