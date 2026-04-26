"""Download the Breast Cancer Wisconsin Diagnostic dataset."""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Project root is two levels above this file (src/data/download.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_CSV_PATH = RAW_DATA_DIR / "breast_cancer.csv"


def _download_from_ucimlrepo() -> pd.DataFrame:
    """Attempt to download dataset via ucimlrepo (dataset ID 17)."""
    from ucimlrepo import fetch_ucirepo  # noqa: PLC0415

    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = dataset.data.targets

    # UCI target column may be 'Diagnosis' with values 'M'/'B'
    if hasattr(y, "columns"):
        target_col = y.columns[0]
        y = y[target_col]
    else:
        y = y.iloc[:, 0]

    df = X.copy()
    df["diagnosis"] = y.values
    return df


def _load_from_sklearn() -> pd.DataFrame:
    """Load the dataset from scikit-learn and return as DataFrame."""
    from sklearn.datasets import load_breast_cancer  # noqa: PLC0415

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    # sklearn: 0=malignant, 1=benign — store as-is, label with 'M'/'B' for uniformity
    df["diagnosis"] = ["M" if t == 0 else "B" for t in data.target]
    return df


def download_data(force: bool = False) -> Path:
    """Download raw data; returns path to saved CSV.

    Priority:
    1. ucimlrepo
    2. sklearn fallback

    Args:
        force: Re-download even if file already exists.

    Returns:
        Path to the saved CSV file.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_CSV_PATH.exists() and not force:
        logger.info("Raw data already exists at %s — skipping download.", RAW_CSV_PATH)
        print(f"[download] Raw data already present: {RAW_CSV_PATH}")
        return RAW_CSV_PATH

    df: pd.DataFrame | None = None

    # --- Try ucimlrepo first ---
    try:
        print("[download] Attempting download via ucimlrepo …")
        df = _download_from_ucimlrepo()
        print("[download] Successfully fetched dataset from UCI ML Repository.")
        logger.info("Dataset downloaded via ucimlrepo.")
    except Exception as exc:  # noqa: BLE001
        print(f"[download] ucimlrepo failed ({exc}); falling back to sklearn.")
        logger.warning("ucimlrepo download failed: %s", exc)

    # --- Fallback: sklearn ---
    if df is None:
        try:
            print("[download] Loading dataset from sklearn …")
            df = _load_from_sklearn()
            print("[download] Successfully loaded dataset from sklearn.")
            logger.info("Dataset loaded via sklearn fallback.")
        except Exception as exc:  # noqa: BLE001
            logger.error("sklearn fallback failed: %s", exc)
            raise RuntimeError(
                "Could not obtain the Breast Cancer dataset from either ucimlrepo or sklearn."
            ) from exc

    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"[download] Raw data saved to {RAW_CSV_PATH} ({len(df)} rows, {df.shape[1]} columns).")
    logger.info("Saved raw data to %s", RAW_CSV_PATH)
    return RAW_CSV_PATH
