"""Page 3: Feature Engineering — Pipeline flow, leakage prevention, engineered features."""

import streamlit as st

from app.utils import (
    load_metadata,
    section_header,
)


def render():
    st.markdown("# ⚙️ Feature Engineering")
    st.caption("Leakage-free preprocessing pipeline built as a scikit-learn transformer.")

    meta = load_metadata()
    fe_meta = meta.get("feature_engineering", {})

    # ── Feature count pipeline ─────────────────────────────────────────────────
    section_header("Pipeline Stages & Feature Counts")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Raw application features", "122")
    with c2:
        st.metric("After engineered features", "~130", "+8 ratio/interaction features")
    with c3:
        st.metric("After historical join", "~190", "+60 bureau/POS/CC/instal aggregates")
    with c4:
        st.metric(
            "Final model features", str(fe_meta.get("n_final_features", 250)), "After preprocessing"
        )

    # ── Pipeline flow diagram ──────────────────────────────────────────────────
    section_header("Pipeline Architecture")
    st.markdown("""
```
application_train.csv (122 cols)
        │
        ▼
FullFeatureEngineering.fit(X_train)          ← fit() computes bin edges from training data ONLY
 ├─ DAYS_EMPLOYED anomaly fix (365243 → NaN)
 ├─ Ratio features: PAYMENT_RATE, CREDIT_INCOME_PERCENT, ANNUITY_INCOME_PERCENT, YEARS_BIRTH
 ├─ Interaction: EXT_SOURCE_PRODUCT = EXT_SOURCE_1 × EXT_SOURCE_2 × EXT_SOURCE_3
 ├─ Quantile bins: np.nanquantile() on X_train → store bin_edges_
 └─ Historical join: merge pre-built Parquet cache on SK_ID_CURR

FullFeatureEngineering.transform(X_val / X_test / inference)
 ├─ Apply the SAME bin edges from fit() — no recomputation ← LEAKAGE FIX
 └─ Historical join: same cached Parquet (no re-aggregation)
        │
        ▼
ColumnTransformer (preprocessor)
 ├─ Numeric → PowerTransformer + StandardScaler
 └─ Categorical → OneHotEncoder (handle_unknown='ignore')
        │
        ▼
LGBMClassifier (766 estimators, Optuna-tuned)
```
    """)

    # ── Engineered features table ──────────────────────────────────────────────
    section_header("Engineered Features")
    tab1, tab2 = st.tabs(["Application Table Features", "Historical Aggregates"])

    with tab1:
        st.markdown("""
| Feature | Formula | Rationale |
|---|---|---|
| `PAYMENT_RATE` | AMT_ANNUITY / AMT_CREDIT | Monthly burden relative to total credit |
| `CREDIT_INCOME_PERCENT` | AMT_CREDIT / AMT_INCOME_TOTAL | Affordability: credit vs income |
| `ANNUITY_INCOME_PERCENT` | AMT_ANNUITY / AMT_INCOME_TOTAL | Monthly payment burden |
| `YEARS_BIRTH` | DAYS_BIRTH / -365 | Age in years (more interpretable) |
| `EXT_SOURCE_PRODUCT` | EXT_SOURCE_1 × EXT_SOURCE_2 × EXT_SOURCE_3 | Multiplicative interaction captures joint low scores |
| `DAYS_EMPLOYED_ANOMALY` | 1 if DAYS_EMPLOYED == 365243 | Sentinel value indicating unemployed |
| `INCOME_QUANTILE_BINS` | `pd.cut(income, bin_edges_['INCOME_QUANTILE_BINS'])` | Quantile bins from training data |
| `CREDIT_QUANTILE_BINS` | `pd.cut(credit, bin_edges_['CREDIT_QUANTILE_BINS'])` | Quantile bins from training data |
        """)

    with tab2:
        st.markdown("""
Six historical tables are aggregated at `SK_ID_CURR` level **offline** (once) and cached as Parquet.
Key aggregates:

| Table | Key Features |
|---|---|
| `bureau.csv` | `BUREAU_DAYS_CREDIT_MAX/MIN`, `BUREAU_AMT_CREDIT_SUM_DEBT_MEAN`, overdue amounts |
| `bureau_balance.csv` | Monthly status distribution (C, X, 1–5 DPD flags) |
| `previous_application.csv` | `PREV_AMT_ANNUITY_MEAN`, approval rates, contract types |
| `installments_payments.csv` | `INSTAL_PAYMENT_PERC_MEAN`, `INSTAL_PAYMENT_DIFF_MEAN`, shortfall indicators |
| `POS_CASH_balance.csv` | `POS_SK_DPD_MAX`, `POS_CNT_INSTALMENT_FUTURE_MEAN` |
| `credit_card_balance.csv` | Utilisation ratios, DPD, balance/limit ratios |
        """)

    # ── Leakage prevention ────────────────────────────────────────────────────
    section_header("Leakage Prevention")
    st.markdown("""
| Transformation | Old (Leaky) Implementation | Fixed Implementation |
|---|---|---|
| Quantile binning | `pd.qcut()` in `transform()` — recomputes from test/inference data | `np.nanquantile()` in `fit()` → `pd.cut()` with stored edges in `transform()` |
| Historical aggregation | Called raw CSVs inside `transform()` — reads GBs per API request | Pre-computed Parquet loaded once; join-only in `transform()` |
| Imputation | Fresh `SimpleImputer` per call | Imputer fitted once on full historical tables |
| Feature selection | Static list from notebook | JSON files generated from validation-set selection |
    """)

    st.success(
        "✅ **Verified by test suite**: `tests/test_feature_engineering.py` "
        "confirms bin edges are NOT recomputed on transform data using a deliberately "
        "skewed test distribution (10× income scale)."
    )

    # ── Bin edges visualization ────────────────────────────────────────────────
    section_header("Temporal Leakage in Bureau Data")
    st.warning("""
**Known limitation (documented):** Some `DAYS_CREDIT` values in `bureau.csv` are positive,
meaning the bureau record postdates the current loan application. A strict production system
would exclude records with `DAYS_CREDIT > 0`. This is documented as a known limitation
and tracked in the Future Improvements section of the README.
    """)
