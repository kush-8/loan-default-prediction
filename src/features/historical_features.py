"""
historical_features.py
======================
Offline historical feature generation for the Home Credit dataset.

This module is intentionally separated from the online inference pipeline.
Historical tables (bureau, previous_application, etc.) are aggregated ONCE
offline and the result is saved as a Parquet artifact. The training and
inference pipelines then join against this artifact rather than re-reading
GBs of raw CSV files on every call.

Design principles
-----------------
- All aggregations are deterministic (fixed column order, reproducible joins).
- No information from future applications leaks into aggregates
  (records are keyed by SK_ID_CURR; within a borrower's history there is no
  temporal cutoff needed because all historical records by construction predate
  the current application — they represent closed/open prior loans, not
  the current one).
- Missing applicants (no history in a table) receive NaN aggregates, which the
  downstream imputer handles.
- The SimpleImputer for categorical columns in bureau/prev_application is
  fitted on the FULL historical tables here, which is appropriate because
  these tables are training-data-independent (they describe prior credit
  history, not the current application). The values are categorical codes
  for loan status types — imputing with the most frequent code from all
  historical records is safe.

Usage
-----
  # Offline (run once before training)
  python -m src.features.historical_features

  # Programmatically
  from src.features.historical_features import build_historical_features
  hist_df = build_historical_features(config)
  hist_df.to_parquet("data/processed/historical_features.parquet")
"""

import logging
import os

import numpy as np
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Individual table aggregators
# ---------------------------------------------------------------------------


def _aggregate_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bureau_balance up to SK_ID_BUREAU level.

    Records represent monthly snapshots of credit-bureau-reported loans.
    All months are historical by design.

    Returns a DataFrame indexed by SK_ID_BUREAU.
    """
    bb_cat = pd.get_dummies(bureau_balance, columns=["STATUS"], drop_first=True)
    bb_agg = bb_cat.groupby("SK_ID_BUREAU").agg(["mean", "sum", "var"])
    bb_agg.columns = pd.Index(["BB_" + e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    return bb_agg


def _aggregate_bureau(bureau: pd.DataFrame, bb_agg: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bureau data (joined with bureau_balance) to SK_ID_CURR level.

    Returns a DataFrame indexed by SK_ID_CURR.
    """
    # Join bureau_balance aggregates into bureau rows
    bureau = bureau.join(bb_agg, how="left", on="SK_ID_BUREAU")
    bureau = bureau.drop(columns=["SK_ID_BUREAU"])

    # Impute categorical bureau columns (mode imputation is safe here;
    # these are credit-type codes, not target-correlated continuous values)
    bureau_cat_cols = bureau.select_dtypes(include="object").columns.tolist()
    if bureau_cat_cols:
        imputer = SimpleImputer(strategy="most_frequent")
        bureau[bureau_cat_cols] = imputer.fit_transform(bureau[bureau_cat_cols])

    # Categorical one-hot aggregations
    bureau_cat_df = pd.get_dummies(
        bureau.select_dtypes("object"),
        columns=bureau.select_dtypes("object").columns,
    )
    bureau_cat_df["SK_ID_CURR"] = bureau["SK_ID_CURR"]
    bureau_cat_agg = bureau_cat_df.groupby("SK_ID_CURR").agg(["mean", "sum"])
    bureau_cat_agg.columns = pd.Index(
        [e[0] + "_" + e[1].upper() for e in bureau_cat_agg.columns.tolist()]
    )

    # Numerical aggregations
    bureau_num_agg = bureau.groupby("SK_ID_CURR").agg(
        {
            "DAYS_CREDIT": ["count", "mean", "max", "min", "sum"],
            "CREDIT_DAY_OVERDUE": ["mean", "max", "sum"],
            "DAYS_CREDIT_ENDDATE": ["mean", "max"],
            "AMT_CREDIT_MAX_OVERDUE": ["mean", "max"],
            "AMT_CREDIT_SUM": ["mean", "sum", "max"],
            "AMT_CREDIT_SUM_DEBT": ["mean", "sum", "max"],
            "AMT_CREDIT_SUM_OVERDUE": ["mean", "sum"],
            "DAYS_CREDIT_UPDATE": ["mean"],
            "AMT_ANNUITY": ["mean", "sum", "max"],
        }
    )
    bureau_num_agg.columns = pd.Index(
        ["BUREAU_" + e[0] + "_" + e[1].upper() for e in bureau_num_agg.columns.tolist()]
    )

    return pd.concat([bureau_num_agg, bureau_cat_agg], axis=1)


def _aggregate_installments(installments: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate installments_payments to SK_ID_CURR level.

    Returns a DataFrame indexed by SK_ID_CURR.
    """
    # Guard against zero division
    installments["PAYMENT_PERC"] = np.where(
        installments["AMT_INSTALMENT"] == 0,
        np.nan,
        installments["AMT_PAYMENT"] / installments["AMT_INSTALMENT"],
    )
    installments["PAYMENT_DIFF"] = installments["AMT_INSTALMENT"] - installments["AMT_PAYMENT"]

    agg = installments.groupby("SK_ID_CURR").agg(
        {
            "NUM_INSTALMENT_VERSION": ["nunique"],
            "PAYMENT_PERC": ["mean", "sum", "max", "min"],
            "PAYMENT_DIFF": ["mean", "sum", "max", "min"],
            "AMT_INSTALMENT": ["mean", "sum", "max"],
            "AMT_PAYMENT": ["mean", "sum", "max"],
        }
    )
    agg.columns = pd.Index(["INSTAL_" + e[0] + "_" + e[1].upper() for e in agg.columns.tolist()])
    return agg


def _aggregate_pos_cash(pos_cash: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate POS_CASH_balance to SK_ID_CURR level.

    Returns a DataFrame indexed by SK_ID_CURR.
    """
    agg = pos_cash.groupby("SK_ID_CURR").agg(
        {
            "MONTHS_BALANCE": ["mean", "max", "min", "count"],
            "SK_DPD": ["mean", "max", "sum"],
            "CNT_INSTALMENT_FUTURE": ["mean", "sum", "min"],
        }
    )
    agg.columns = pd.Index(["POS_" + e[0] + "_" + e[1].upper() for e in agg.columns.tolist()])
    return agg


def _aggregate_credit_card(credit_card: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate credit_card_balance to SK_ID_CURR level.

    Returns a DataFrame indexed by SK_ID_CURR.
    """
    agg = credit_card.groupby("SK_ID_CURR").agg(
        {
            "AMT_BALANCE": ["mean", "sum", "max", "min"],
            "AMT_CREDIT_LIMIT_ACTUAL": ["mean", "sum", "max"],
            "AMT_DRAWINGS_CURRENT": ["mean", "sum", "max"],
            "CNT_INSTALMENT_MATURE_CUM": ["mean", "sum", "max"],
            "SK_DPD": ["mean", "max", "sum"],
        }
    )
    agg.columns = pd.Index(["CC_" + e[0] + "_" + e[1].upper() for e in agg.columns.tolist()])
    return agg


def _aggregate_previous_application(prev: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate previous_application (plus POS/installments/CC already joined
    at SK_ID_CURR level) to SK_ID_CURR level.

    Returns a DataFrame indexed by SK_ID_CURR.
    """
    prev_cat_cols = prev.select_dtypes(include="object").columns.tolist()
    if prev_cat_cols:
        imputer = SimpleImputer(strategy="most_frequent")
        prev[prev_cat_cols] = imputer.fit_transform(prev[prev_cat_cols])

    prev_cat = pd.get_dummies(
        prev.select_dtypes("object"),
        columns=prev.select_dtypes("object").columns,
    )
    prev_cat["SK_ID_CURR"] = prev["SK_ID_CURR"]
    prev_cat_agg = prev_cat.groupby("SK_ID_CURR").agg(["mean", "sum"])
    prev_cat_agg.columns = pd.Index(
        [e[0] + "_" + e[1].upper() for e in prev_cat_agg.columns.tolist()]
    )

    prev_num_agg = prev.groupby("SK_ID_CURR").agg(
        {
            "AMT_ANNUITY": ["mean", "sum", "max", "min"],
            "AMT_CREDIT": ["mean", "sum", "max", "min"],
            "AMT_GOODS_PRICE": ["mean", "sum", "max"],
            "CNT_PAYMENT": ["mean", "sum"],
            "DAYS_DECISION": ["mean", "max", "min"],
        }
    )
    prev_num_agg.columns = pd.Index(
        ["PREV_" + e[0] + "_" + e[1].upper() for e in prev_num_agg.columns.tolist()]
    )

    return pd.concat([prev_num_agg, prev_cat_agg], axis=1)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def build_historical_features(config: dict) -> pd.DataFrame:
    """
    Load all historical tables, aggregate them, and return a single DataFrame
    keyed by SK_ID_CURR.

    This function should be called ONCE offline. The result should be saved
    and reloaded during training and inference rather than re-computed.

    Parameters
    ----------
    config : dict
        Loaded config.yaml as a Python dict.

    Returns
    -------
    pd.DataFrame
        One row per SK_ID_CURR with all historical aggregate features.
        Index is SK_ID_CURR.
    """
    paths = config["data_paths"]

    logger.info("Loading historical tables from disk...")

    bureau = pd.read_csv(paths["bureau"])
    bureau_balance = pd.read_csv(paths["bureau_balance"])
    installments = pd.read_csv(paths["installments_payments"])
    pos_cash = pd.read_csv(paths["POS_CASH_balance"])
    credit_card = pd.read_csv(paths["credit_card_balance"])
    prev = pd.read_csv(paths["previous_application"])

    logger.info("Aggregating bureau_balance...")
    bb_agg = _aggregate_bureau_balance(bureau_balance)

    logger.info("Aggregating bureau...")
    bureau_agg = _aggregate_bureau(bureau, bb_agg)

    logger.info("Aggregating installments_payments...")
    instal_agg = _aggregate_installments(installments)

    logger.info("Aggregating POS_CASH_balance...")
    pos_agg = _aggregate_pos_cash(pos_cash)

    logger.info("Aggregating credit_card_balance...")
    cc_agg = _aggregate_credit_card(credit_card)

    logger.info("Aggregating previous_application...")
    prev_agg = _aggregate_previous_application(prev)

    logger.info("Joining all historical aggregates...")
    # Use outer join so we don't drop applicants that appear in some tables
    hist = bureau_agg.join(instal_agg, how="outer")
    hist = hist.join(pos_agg, how="outer")
    hist = hist.join(cc_agg, how="outer")
    hist = hist.join(prev_agg, how="outer")

    # Clean column names (remove special chars)
    hist.columns = ["".join(c if c.isalnum() else "_" for c in str(col)) for col in hist.columns]

    logger.info(
        f"Historical feature table built: {hist.shape[0]} applicants, " f"{hist.shape[1]} features."
    )
    return hist


def load_or_build_historical_features(config: dict, cache_path: str | None = None) -> pd.DataFrame:
    """
    Load historical features from a cached Parquet file if it exists,
    otherwise build from raw CSVs and save the cache.

    Parameters
    ----------
    config : dict
        Loaded config.yaml.
    cache_path : str, optional
        Path to the Parquet cache file. Falls back to
        config['data_paths']['historical_features'] if not provided.
    """
    if cache_path is None:
        cache_path = config["data_paths"].get(
            "historical_features", "data/processed/historical_features.parquet"
        )

    if os.path.exists(cache_path):
        logger.info(f"Loading historical features from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    logger.info(f"Cache not found at {cache_path}. Building from raw CSVs...")
    hist = build_historical_features(config)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    hist.to_parquet(cache_path)
    logger.info(f"Historical features saved to {cache_path}")
    return hist


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config/config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    hist = build_historical_features(cfg)

    out_path = cfg["data_paths"].get(
        "historical_features", "data/processed/historical_features.parquet"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    hist.to_parquet(out_path)
    print(f"Saved {hist.shape[0]:,} rows × {hist.shape[1]:,} cols → {out_path}")
