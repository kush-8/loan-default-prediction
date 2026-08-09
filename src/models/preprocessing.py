"""
preprocessing.py
================
Sklearn-compatible feature engineering transformer and preprocessing pipeline
factory for the Home Credit Default Risk dataset.

Design principles
-----------------
1. **Leakage-free**: All data-dependent operations (quantile bin edges,
   imputation parameters) are fitted ONLY on training data inside ``fit()``
   and applied without recomputation in ``transform()``.

2. **Stateful**: ``FullFeatureEngineering`` stores all learned parameters as
   instance attributes (sklearn convention: attribute names end with ``_``).

3. **Separation of concerns**:
   - ``FullFeatureEngineering`` handles application-table feature creation
     and join of pre-computed historical features.
   - Historical aggregation lives in ``src/features/historical_features.py``
     and runs offline (not on every predict call).
   - ``create_preprocessor`` builds the sklearn ColumnTransformer for
     imputation, scaling, and encoding.

4. **Train / Validation / Test contract**:
   TRAIN:     raw → FullFeatureEngineering.fit_transform()
                  → Preprocessor.fit_transform()
                  → Classifier.fit()
   VAL/TEST:  raw → FullFeatureEngineering.transform()
                  → Preprocessor.transform()
                  → Classifier.predict_proba()
   INFERENCE: same as VAL/TEST, using the serialised pipeline from joblib.

Usage
-----
See train.py for the full pipeline assembly.
"""

import json
import logging
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SPECIAL_DAYS_EMPLOYED_VALUE = 365243  # encodes "pensioner / not working"
_N_QUANTILE_BINS = 5


def _clean_col_names(df: pd.DataFrame) -> pd.DataFrame:
    """Replace non-alphanumeric characters in column names with underscores."""
    df.columns = ["".join(c if c.isalnum() else "_" for c in str(col)) for col in df.columns]
    return df


# ---------------------------------------------------------------------------
# Main sklearn Transformer
# ---------------------------------------------------------------------------


class FullFeatureEngineering(BaseEstimator, TransformerMixin):
    """
    End-to-end feature engineering transformer for the application table.

    Responsibilities
    ----------------
    - Flag and replace the anomalous DAYS_EMPLOYED sentinel value.
    - Derive ratio and time-based features.
    - Derive the EXT_SOURCE_PRODUCT interaction feature.
    - Bin continuous features into quantile groups using bin edges computed
      on training data only (stored in ``fit()``).
    - Join pre-computed historical features from an offline artifact.
    - Select the final feature set.

    Parameters
    ----------
    config_path : str
        Path to config/config.yaml (relative to working directory).
    historical_features_df : pd.DataFrame, optional
        Pre-loaded historical feature DataFrame indexed by SK_ID_CURR.
        If None, the transformer will attempt to load it from the path
        specified in the config. Pass this explicitly in tests to avoid
        file I/O.

    Fitted attributes (set in fit(), sklearn convention)
    ---------------------------------------------------
    bin_edges_ : dict[str, np.ndarray]
        Quantile bin edges computed on X_train for each binned column.
    main_table_features_ : list[str]
        Application-table feature columns retained after first selection.
    final_features_ : list[str]
        Final feature columns after historical join.
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        historical_features_df: pd.DataFrame | None = None,
    ) -> None:
        self.config_path = config_path
        self.historical_features_df = historical_features_df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        with open(self.config_path) as fh:
            return yaml.safe_load(fh)

    def _load_feature_lists(self, config: dict) -> tuple:
        """Return (main_table_top_features, final_top_features)."""
        with open(config["model_assets"]["main_table_top_features"]) as fh:
            main_feats = json.load(fh)
        with open(config["model_assets"]["final_top_features"]) as fh:
            final_feats = json.load(fh)
        return main_feats, final_feats

    def _get_historical_df(self, config: dict) -> pd.DataFrame:
        """Return the pre-computed historical feature DataFrame."""
        if self.historical_features_df is not None:
            return self.historical_features_df

        cache_path = config["data_paths"].get(
            "historical_features", "data/processed/historical_features.parquet"
        )
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Historical features cache not found at '{cache_path}'. "
                "Run: python -m src.features.historical_features"
            )
        logger.info(f"Loading historical features from: {cache_path}")
        return pd.read_parquet(cache_path)

    # ------------------------------------------------------------------
    # Core feature engineering (application table only)
    # ------------------------------------------------------------------

    @staticmethod
    def _engineer_application_features(X: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from the application table.
        All operations here are deterministic given X — no statistics
        from training data are used.
        """
        X = X.copy()

        # Anomaly flags and fixes
        days_employed = (
            X["DAYS_EMPLOYED"] if "DAYS_EMPLOYED" in X.columns else pd.Series(np.nan, index=X.index)
        )
        X["DAYS_EMPLOYED_ANOMALY"] = (days_employed == _SPECIAL_DAYS_EMPLOYED_VALUE).astype(int)
        X["DAYS_EMPLOYED"] = days_employed.replace({_SPECIAL_DAYS_EMPLOYED_VALUE: np.nan})

        own_car_age = (
            X["OWN_CAR_AGE"] if "OWN_CAR_AGE" in X.columns else pd.Series(np.nan, index=X.index)
        )
        X["FLAG_OWN_CAR"] = own_car_age.notna().astype(int)

        # Time-based features
        days_birth = (
            X["DAYS_BIRTH"] if "DAYS_BIRTH" in X.columns else pd.Series(np.nan, index=X.index)
        )
        X["YEARS_BIRTH"] = days_birth / -365.0

        # Ratio features — guard against zero denominators
        amt_income = (
            X["AMT_INCOME_TOTAL"]
            if "AMT_INCOME_TOTAL" in X.columns
            else pd.Series(np.nan, index=X.index)
        )
        amt_credit = (
            X["AMT_CREDIT"] if "AMT_CREDIT" in X.columns else pd.Series(np.nan, index=X.index)
        )
        amt_annuity = (
            X["AMT_ANNUITY"] if "AMT_ANNUITY" in X.columns else pd.Series(np.nan, index=X.index)
        )

        income = amt_income.replace(0, np.nan)
        credit = amt_credit.replace(0, np.nan)

        X["CREDIT_INCOME_PERCENT"] = amt_credit / income
        X["ANNUITY_INCOME_PERCENT"] = amt_annuity / income
        X["PAYMENT_RATE"] = amt_annuity / credit

        # Interaction feature — safe against missing EXT_SOURCE columns
        src1 = (
            X["EXT_SOURCE_1"] if "EXT_SOURCE_1" in X.columns else pd.Series(np.nan, index=X.index)
        )
        src2 = (
            X["EXT_SOURCE_2"] if "EXT_SOURCE_2" in X.columns else pd.Series(np.nan, index=X.index)
        )
        src3 = (
            X["EXT_SOURCE_3"] if "EXT_SOURCE_3" in X.columns else pd.Series(np.nan, index=X.index)
        )
        X["EXT_SOURCE_PRODUCT"] = src1 * src2 * src3

        return X

    # ------------------------------------------------------------------
    # sklearn fit / transform
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "FullFeatureEngineering":
        """
        Learn all data-dependent parameters from training data.

        Parameters learned
        ------------------
        - Quantile bin edges for 4 continuous features (using np.nanquantile
          on training data only).
        - Feature lists from config JSON files.
        """
        config = self._load_config()
        main_feats, final_feats = self._load_feature_lists(config)

        # Engineer features first so we compute bins on the engineered columns
        X_eng = self._engineer_application_features(X)

        # Compute quantile bin edges from training data only
        # We use np.nanquantile so NaN rows don't corrupt edge computation.
        self.bin_edges_: dict = {}
        bin_cols = {
            "INCOME_QUANTILE_BINS": "AMT_INCOME_TOTAL",
            "CREDIT_QUANTILE_BINS": "AMT_CREDIT",
            "GOODS_PRICE_QUANTILE_BINS": "AMT_GOODS_PRICE",
            "ANNUITY_QUANTILE_BINS": "AMT_ANNUITY",
        }
        quantiles = np.linspace(0, 1, _N_QUANTILE_BINS + 1)
        for bin_col, src_col in bin_cols.items():
            if src_col in X_eng.columns:
                edges = np.nanquantile(X_eng[src_col].values, quantiles)
                # Ensure uniqueness (handles degenerate distributions)
                edges = np.unique(edges)
                self.bin_edges_[bin_col] = edges
                logger.debug(f"Bin edges for {bin_col}: {edges}")

        # Store feature lists
        self.main_table_features_ = main_feats
        self.final_features_ = final_feats

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature engineering transformations.

        Bin edges computed in ``fit()`` are reused here — they are NOT
        recomputed from X.
        """
        if not hasattr(self, "bin_edges_"):
            raise RuntimeError(
                "FullFeatureEngineering has not been fitted. " "Call fit() before transform()."
            )

        config = self._load_config()
        X_copy = self._engineer_application_features(X)

        # Apply pre-computed quantile bins using pd.cut (NOT pd.qcut)
        bin_cols = {
            "INCOME_QUANTILE_BINS": "AMT_INCOME_TOTAL",
            "CREDIT_QUANTILE_BINS": "AMT_CREDIT",
            "GOODS_PRICE_QUANTILE_BINS": "AMT_GOODS_PRICE",
            "ANNUITY_QUANTILE_BINS": "AMT_ANNUITY",
        }
        for bin_col, src_col in bin_cols.items():
            if src_col in X_copy.columns and bin_col in self.bin_edges_:
                edges = self.bin_edges_[bin_col]
                X_copy[bin_col] = pd.cut(
                    X_copy[src_col],
                    bins=edges,
                    labels=False,
                    include_lowest=True,
                )

        # Select application-table features
        X_copy = _clean_col_names(X_copy)
        # Ensure all required application-table features exist
        for col in self.main_table_features_:
            if col not in X_copy.columns:
                X_copy[col] = np.nan

        keep_cols = ["SK_ID_CURR"] + self.main_table_features_
        if "SK_ID_CURR" not in X_copy.columns:
            X_copy["SK_ID_CURR"] = np.nan
        X_copy = X_copy[keep_cols]

        # Join pre-computed historical features
        hist_df = self._get_historical_df(config)
        X_copy = X_copy.join(hist_df, how="left", on="SK_ID_CURR")

        # Replace infinities introduced by ratio features
        X_copy.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Clean column names again after join
        X_copy = _clean_col_names(X_copy)

        # Select final feature set
        available_final = [col for col in self.final_features_ if col in X_copy.columns]
        keep_final = ["SK_ID_CURR"] + available_final
        keep_final = [c for c in keep_final if c in X_copy.columns]

        return X_copy[keep_final]


# ---------------------------------------------------------------------------
# Preprocessor factory
# ---------------------------------------------------------------------------


def create_preprocessor(numerical_cols: list, categorical_cols: list) -> ColumnTransformer:
    """
    Build and return the sklearn ColumnTransformer for standard preprocessing.

    Pipeline per column type
    -----------------------
    Numerical : median imputation → Yeo-Johnson power transform → standard scale
    Categorical: mode imputation → one-hot encoding (drop first, ignore unknown)

    Parameters
    ----------
    numerical_cols : list[str]
        Column names to treat as numerical.
    categorical_cols : list[str]
        Column names to treat as categorical.

    Returns
    -------
    ColumnTransformer (unfitted)
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Standalone test block
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    with open(config_path) as fh:
        config = yaml.safe_load(fh)

    app_train = pd.read_csv(config["data_paths"]["application_train"])

    fe = FullFeatureEngineering(config_path=config_path)
    df_eng = fe.fit_transform(app_train.drop(columns=["TARGET"]))
    print(f"Shape after feature engineering: {df_eng.shape}")
    print(f"Stored bin edges: {list(fe.bin_edges_.keys())}")

    numerical_cols = [
        c for c in df_eng.select_dtypes(include=np.number).columns if c != "SK_ID_CURR"
    ]
    categorical_cols = df_eng.select_dtypes(include="object").columns.tolist()

    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    X = df_eng.drop(columns=["SK_ID_CURR"])
    X_processed = preprocessor.fit_transform(X)
    print(f"Shape after preprocessing: {X_processed.shape}")
    print("Preprocessing test completed successfully.")
