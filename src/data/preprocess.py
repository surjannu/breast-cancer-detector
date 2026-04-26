"""Preprocess the raw breast cancer CSV into train/test splits."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandas.api.types as pat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "breast_cancer.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def _encode_target(series: pd.Series) -> pd.Series:
    """Encode diagnosis column to numeric: Malignant=1, Benign=0.

    Handles both string labels ('M'/'B') and numeric sklearn labels (0=M,1=B).
    """
    if not pat.is_numeric_dtype(series):
        # String labels from UCI: 'M' → 1, 'B' → 0
        mapping = {"M": 1, "B": 0}
        encoded = series.str.strip().map(mapping)
        if encoded.isna().any():
            raise ValueError(
                f"Unexpected diagnosis values: {series.unique()}. Expected 'M' or 'B'."
            )
        return encoded.astype(int)
    else:
        # Numeric labels from sklearn: 0=malignant → 1, 1=benign → 0
        # sklearn target_names = ['malignant', 'benign']  (0=malignant, 1=benign)
        # We re-encode so 1=Malignant, 0=Benign for consistent medical convention
        return series.map({0: 1, 1: 0}).astype(int)


def preprocess_data(
    raw_csv: Path | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """Load, clean, encode, scale and split data.

    Returns:
        X_train, X_test, y_train, y_test, feature_names (list), scaler
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if raw_csv is None:
        raw_csv = RAW_CSV_PATH

    print(f"[preprocess] Loading raw data from {raw_csv} …")
    df = pd.read_csv(raw_csv)

    # Drop any unnamed index columns that sometimes appear
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]

    print(f"[preprocess] Raw shape: {df.shape}")
    logger.info("Loaded raw data: %s", df.shape)

    # ---- Missing values ----
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"[preprocess] Handling {missing} missing values (column-wise median imputation).")
        df = df.fillna(df.median(numeric_only=True))
    else:
        print("[preprocess] No missing values found.")

    # ---- Duplicates ----
    dupes = df.duplicated().sum()
    if dupes > 0:
        print(f"[preprocess] Removing {dupes} duplicate rows.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print("[preprocess] No duplicate rows found.")

    # ---- Target encoding ----
    if "diagnosis" not in df.columns:
        raise KeyError("Expected a 'diagnosis' column in the raw CSV.")

    df["diagnosis"] = _encode_target(df["diagnosis"])

    # ---- Separate features / target ----
    y = df["diagnosis"]
    X = df.drop(columns=["diagnosis"])
    feature_names = list(X.columns)

    print(f"[preprocess] Features: {len(feature_names)}  |  Samples: {len(y)}")
    print(f"[preprocess] Class distribution — Malignant (1): {(y == 1).sum()}  |  Benign (0): {(y == 0).sum()}")

    # ---- Scaling ----
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

    # ---- Train / Test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled_df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(f"[preprocess] Train size: {len(X_train)}  |  Test size: {len(X_test)}")
    logger.info("Split — train=%d, test=%d", len(X_train), len(X_test))

    # ---- Save processed data ----
    X_train.to_csv(PROCESSED_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DIR / "y_test.csv", index=False)

    # Save feature names and scaler
    pd.Series(feature_names).to_csv(PROCESSED_DIR / "feature_names.csv", index=False, header=["feature"])
    joblib.dump(scaler, PROCESSED_DIR / "scaler.pkl")

    # Save full cleaned (unscaled) dataset for EDA
    df_clean = pd.DataFrame(X.values, columns=feature_names)
    df_clean["diagnosis"] = y.values
    df_clean.to_csv(PROCESSED_DIR / "breast_cancer_clean.csv", index=False)

    print(f"[preprocess] Processed data saved to {PROCESSED_DIR}")
    logger.info("Preprocessing complete.")

    return (
        X_train.values,
        X_test.values,
        y_train.values,
        y_test.values,
        feature_names,
        scaler,
    )


def load_processed_data() -> tuple:
    """Load already-processed data from disk.

    Returns:
        X_train, X_test, y_train, y_test, feature_names (list), scaler
    """
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv").values
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv").values
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()
    feature_names = pd.read_csv(PROCESSED_DIR / "feature_names.csv")["feature"].tolist()
    scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")
    return X_train, X_test, y_train, y_test, feature_names, scaler
