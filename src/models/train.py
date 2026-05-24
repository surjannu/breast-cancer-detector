"""Train multiple classifiers on the breast cancer dataset."""

import logging
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def build_models() -> dict:
    """Return a dict of {name: base estimator} for all classifiers to train."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
        ),
        "KNN": KNeighborsClassifier(
            n_jobs=-1,
        ),
    }


def build_param_grids() -> dict:
    """Return hyperparameter search grids for GridSearchCV."""
    return {
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "solver": ["lbfgs", "liblinear"],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
        },
        "SVM": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", "auto"],
        },
        "XGBoost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
        },
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "weights": ["uniform", "distance"],
            "metric": ["euclidean", "manhattan"],
        },
    }


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_dir: Path | None = None,
) -> dict:
    """Apply SMOTE, tune hyperparameters via GridSearchCV, and persist models.

    Args:
        X_train: Training features (already scaled).
        y_train: Training labels.
        models_dir: Directory to save .pkl files (defaults to PROJECT_ROOT/models).

    Returns:
        Dict of {model_name: fitted GridSearchCV estimator}.
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Apply SMOTE to address class imbalance ─────────────────────────────────
    print("[train] Applying SMOTE to balance training classes …")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    unique, counts = np.unique(y_resampled, return_counts=True)
    class_counts = dict(zip(unique.tolist(), counts.tolist()))
    logger.info(
        "SMOTE: %d → %d samples  class counts: %s",
        len(y_train), len(y_resampled), class_counts,
    )
    print(f"[train]   → Original: {len(y_train)} samples  |  After SMOTE: {len(y_resampled)} samples  |  Classes: {class_counts}")

    models = build_models()
    param_grids = build_param_grids()
    trained: dict = {}

    for name, clf in models.items():
        print(f"[train] Tuning & training {name} …")
        logger.info("GridSearchCV tuning %s", name)
        try:
            grid_search = GridSearchCV(
                estimator=clf,
                param_grid=param_grids[name],
                scoring="f1",
                cv=5,
                n_jobs=-1,
                refit=True,
            )
            grid_search.fit(X_resampled, y_resampled)
            trained[name] = grid_search

            print(f"[train]   → Best params : {grid_search.best_params_}")
            print(f"[train]   → Best CV F1  : {grid_search.best_score_:.4f}")
            logger.info(
                "Best params for %s: %s  (CV F1=%.4f)",
                name, grid_search.best_params_, grid_search.best_score_,
            )

            safe_name = name.lower().replace(" ", "_")
            save_path = models_dir / f"{safe_name}.pkl"
            joblib.dump(grid_search, save_path)
            print(f"[train]   → saved to {save_path}")
            logger.info("Saved %s to %s", name, save_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to train %s: %s", name, exc)
            print(f"[train]   ✗ {name} failed: {exc}")

    print(f"[train] Trained {len(trained)}/{len(models)} models.")
    return trained
