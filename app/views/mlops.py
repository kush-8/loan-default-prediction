"""Page 8: MLOps Architecture — Pipeline, versioning, CI/CD."""

import streamlit as st

from app.utils import (
    load_metadata,
    section_header,
)


def render():
    st.markdown("# 🔧 MLOps Architecture")
    st.caption("Model versioning, reproducibility, CI/CD, and serving infrastructure.")

    meta = load_metadata()
    env = meta.get("environment", {})
    dataset = meta.get("dataset", {})
    split = meta.get("split_strategy", {})

    # ── Model version card ──────────────────────────────────────────────────────
    section_header("Trained Model — Snapshot")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Model Version", meta.get("model_version", "—"))
    with c2:
        ts = meta.get("training_timestamp", "—")
        st.metric("Training Timestamp", ts[:10] if ts != "—" else "—")
    with c3:
        st.metric("Git SHA", meta.get("git_sha", "—"))
    with c4:
        st.metric("Train duration", f"{meta.get('training_duration_seconds', 0):.0f}s")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Python version", env.get("python_version", "—"))
    with c2:
        st.metric("scikit-learn", env.get("sklearn_version", "—"))
    with c3:
        st.metric("LightGBM", env.get("lightgbm_version", "—"))
    with c4:
        st.metric("Platform", env.get("platform", "—")[:15] if env.get("platform") else "—")

    # ── Reproducibility ────────────────────────────────────────────────────────
    section_header("Reproducibility Guarantees")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown(f"""
**Data reproducibility:**
- Dataset hash (MD5, first 10MB): `{dataset.get('hash_md5_first10mb', '—')}`
- Training rows: {dataset.get('n_rows', '—'):,}
- Split: {split.get('train_fraction', 0)*100:.0f}/{split.get('val_fraction', 0)*100:.0f}/{split.get('test_fraction', 0)*100:.0f} (stratified)
- Random seed: `{split.get('random_seed', 42)}`
- Stratified: `{split.get('stratified', True)}`

**Pipeline reproducibility:**
- `final_pipeline.joblib` serialized with joblib
- `model_metadata.json` records all hyperparameters, metrics, and thresholds
- Bin edges stored in `FullFeatureEngineering.bin_edges_` (fitted state serialized)
        """)
    with col_r:
        st.markdown("""
**Known limitations:**
- Optuna hyperparameter search used Bayesian sampling with a fixed seed,
  but exact results may vary across Optuna versions
- Historical feature Parquet cache is not committed (too large)
  → must be regenerated with `historical_features.py`
- `EXT_SOURCE_*` features are opaque external scores — not computable
  from raw application data at inference time

**To reproduce training:**
```bash
python scripts/download_and_train.py
# Downloads data, builds Parquet cache, trains, saves pipeline
```
        """)

    # ── Pipeline architecture ──────────────────────────────────────────────────
    section_header("Training Pipeline Architecture")
    st.markdown("""
```
scripts/download_and_train.py
    │
    ├─ Step 1: Download via kagglehub (Home Credit dataset, ~2GB)
    │
    ├─ Step 2: Build historical feature cache
    │          src/features/historical_features.py
    │          → aggregates 6 tables at SK_ID_CURR level
    │          → saves data/processed/historical_features.parquet
    │
    └─ Step 3: src/models/train.py
               ├─ Stratified 70/15/15 split (seed=42)
               ├─ FullFeatureEngineering.fit(X_train)
               │   ├─ Compute quantile bin edges from training data
               │   └─ Join historical Parquet cache
               ├─ ColumnTransformer.fit(X_train_fe)
               ├─ LGBMClassifier.fit(X_train_pp)
               ├─ Threshold selection on X_val (F1-optimal, cost-optimal, recall-constrained)
               ├─ Calibration evaluation on X_val (sigmoid vs isotonic vs none)
               ├─ FINAL evaluation on X_test (once)
               ├─ Assemble sklearn Pipeline (fe → preprocessor → classifier)
               └─ Save:
                   ├─ models/final_pipeline.joblib
                   └─ models/model_metadata.json
```
    """)

    # ── Inference pipeline ─────────────────────────────────────────────────────
    section_header("Inference Pipeline")
    st.markdown("""
```
HTTP POST /v1/predict
    │
    ├─ Pydantic v2 validation → 422 on invalid input (not 500)
    │
    └─ ModelRegistry.predict(input_df)
        ├─ pipeline.predict_proba(input_df)[:, 1]
        │   ├─ FullFeatureEngineering.transform()  ← uses stored bin_edges_
        │   ├─ ColumnTransformer.transform()
        │   └─ LGBMClassifier.predict_proba()
        │
        └─ Response:
            ├─ probability        (float, 6 decimal places)
            ├─ predicted_class    (0 or 1, at stored threshold)
            ├─ risk_category      (LOW / MEDIUM / HIGH / VERY HIGH)
            ├─ threshold          (from model_metadata.json)
            ├─ model_version      (from model_metadata.json)
            └─ explanation        (SHAP-based, on request)
```
    """)

    # ── CI/CD ──────────────────────────────────────────────────────────────────
    section_header("CI/CD Pipeline (GitHub Actions)")
    tab1, tab2 = st.tabs(["ci.yml — Every Push/PR", "e2e_tests.yml — Manual"])
    with tab1:
        st.markdown("""
```yaml
jobs:
  lint-and-test:
    - ruff check src/ tests/              # lint
    - black --check src/ tests/           # format check
    - pytest tests/test_ci.py             # fast unit tests (no data needed)
    - pytest tests/test_feature_engineering.py  # leakage tests
    - python -c "assert config files exist"  # config verification

  docker-build:
    needs: lint-and-test
    - docker build -t loan-default-prediction:ci .
    - docker run -d -p 8000:8000 ...
    - curl /health → 200 or 503

  security-check:
    needs: lint-and-test
    continue-on-error: true
    - pip-audit -r requirements-api.txt  # CVE scan
```
        """)
    with tab2:
        st.markdown("""
```yaml
# e2e_tests.yml — triggered manually
# Requires Kaggle credentials as GitHub Secrets

jobs:
  full-pipeline:
    - Download dataset (KAGGLE_USERNAME, KAGGLE_KEY)
    - python scripts/download_and_train.py
    - pytest tests/ -m e2e --tb=short
    - Verify models/model_metadata.json schema
```
        """)

    # ── Docker ─────────────────────────────────────────────────────────────────
    section_header("Docker Deployment")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
**Image design:**
- Base: `python:3.10-slim` (minimal footprint)
- Non-root user (`appuser:appgroup`) — security best practice
- Only lean API deps (`requirements-api.txt`) — no training packages
- Health check: `urllib.request` to `/health` every 30s

```bash
# Build and run
docker build -t loan-default-prediction .
docker run -d -p 8000:8000 loan-default-prediction
curl http://localhost:8000/health
```
        """)
    with col_r:
        st.markdown("""
**Endpoints:**
| Method | Path | Purpose |
|---|---|---|
| GET | /health | Liveness probe |
| GET | /ready | Readiness probe |
| GET | /model-info | Full metadata |
| POST | /v1/predict | Prediction |

**Lean image dependencies (~250MB):**
- fastapi, uvicorn, pydantic
- scikit-learn, lightgbm, joblib
- pandas, numpy
        """)
