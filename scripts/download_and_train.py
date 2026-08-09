"""
download_and_train.py
=====================
Downloads the Home Credit Default Risk dataset via kagglehub,
symlinks / copies the CSVs into the expected data/raw/ directory,
generates the historical feature cache, and runs training.

Usage
-----
    python scripts/download_and_train.py

Requirements
------------
- kagglehub installed (pip install kagglehub)
- Kaggle credentials in ~/.kaggle/kaggle.json  OR  KAGGLE_USERNAME+KAGGLE_KEY env vars
"""

import os
import shutil
import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Step 1: Download via kagglehub
# ---------------------------------------------------------------------------

def download_dataset() -> Path:
    """Download dataset and return the path to the extracted files."""
    try:
        import kagglehub
    except ImportError:
        logger.error("kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    logger.info("Downloading Home Credit Default Risk dataset via kagglehub...")
    logger.info("(This may take several minutes — dataset is ~2GB)")

    path = kagglehub.dataset_download("megancrenshaw/home-credit-default-risk")
    logger.info(f"Dataset downloaded to: {path}")
    return Path(path)


# ---------------------------------------------------------------------------
# Step 2: Symlink / copy files into data/raw/
# ---------------------------------------------------------------------------

EXPECTED_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "HomeCredit_columns_description.csv",
]


def setup_data_raw(source_dir: Path, raw_dir: Path) -> None:
    """
    Locate each expected CSV file in source_dir (recursively) and
    create a symlink (or copy on Windows if symlinks unavailable)
    in raw_dir.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Build a map of filename -> actual path from the downloaded directory
    available = {}
    for f in source_dir.rglob("*.csv"):
        available[f.name] = f

    logger.info(f"Files found in downloaded dataset: {sorted(available.keys())}")

    missing = []
    for expected in EXPECTED_FILES:
        if expected in available:
            src = available[expected]
            dst = raw_dir / expected
            if dst.exists() or dst.is_symlink():
                logger.info(f"  Already exists: {dst.name}")
                continue
            try:
                dst.symlink_to(src)
                logger.info(f"  Symlinked: {expected} → {src}")
            except (OSError, NotImplementedError):
                # Windows may require admin for symlinks — fall back to copy
                logger.warning(f"  Symlink failed for {expected}, copying instead...")
                shutil.copy2(src, dst)
                logger.info(f"  Copied: {expected} ({src.stat().st_size / 1e6:.0f} MB)")
        else:
            missing.append(expected)

    if missing:
        logger.warning(f"Files not found in download: {missing}")
        logger.warning("Training will proceed but some historical features may be missing.")


# ---------------------------------------------------------------------------
# Step 3: Generate historical feature cache
# ---------------------------------------------------------------------------

def build_historical_features() -> None:
    import yaml
    config_path = PROJECT_ROOT / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    cache_path = config["data_paths"].get(
        "historical_features", "data/processed/historical_features.parquet"
    )
    cache_full = PROJECT_ROOT / cache_path

    if cache_full.exists():
        logger.info(f"Historical feature cache already exists at {cache_full}. Skipping build.")
        return

    logger.info("Building historical feature cache (this takes ~10–20 minutes)...")
    from src.features.historical_features import build_historical_features as _build
    hist = _build(config)
    cache_full.parent.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(cache_full)
    logger.info(f"Cache saved → {cache_full}  ({hist.shape[0]:,} rows × {hist.shape[1]} cols)")


# ---------------------------------------------------------------------------
# Step 4: Run training
# ---------------------------------------------------------------------------

def run_training() -> None:
    logger.info("=" * 60)
    logger.info("Starting training pipeline...")
    from src.models.train import run_training as _train
    _train(config_path=str(PROJECT_ROOT / "config" / "config.yaml"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raw_dir = PROJECT_ROOT / "data" / "raw"

    # Check if data already present (skip download)
    if (raw_dir / "application_train.csv").exists():
        logger.info("application_train.csv already present in data/raw/ — skipping download.")
    else:
        source_dir = download_dataset()
        setup_data_raw(source_dir, raw_dir)

    # Verify required file exists before continuing
    train_file = raw_dir / "application_train.csv"
    if not train_file.exists():
        logger.error(
            f"application_train.csv not found at {train_file}. "
            "Check the download completed successfully."
        )
        sys.exit(1)

    logger.info(f"application_train.csv confirmed: {train_file.stat().st_size / 1e6:.1f} MB")

    # Build historical features
    build_historical_features()

    # Train
    run_training()

    logger.info("=" * 60)
    logger.info("Pipeline complete.")
    logger.info("  Model:    models/final_pipeline.joblib")
    logger.info("  Metadata: models/model_metadata.json")
    logger.info("  Reports:  reports/")
