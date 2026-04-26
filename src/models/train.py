"""Train multiple classifiers on the breast cancer dataset."""

import logging
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"


def build_models() -> dict:
    """Return a dict of {name: estimator} for all classifiers to train."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1,
        ),
    }


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_dir: Path | None = None,
) -> dict:
    """Fit all models and persist them to disk.

    Args:
        X_train: Training features (already scaled).
        y_train: Training labels.
        models_dir: Directory to save .pkl files (defaults to PROJECT_ROOT/models).

    Returns:
        Dict of {model_name: fitted_estimator}.
    """
    if models_dir is None:
        models_dir = MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    models = build_models()
    trained: dict = {}

    for name, clf in models.items():
        print(f"[train] Training {name} …")
        logger.info("Training %s", name)
        try:
            clf.fit(X_train, y_train)
            trained[name] = clf

            safe_name = name.lower().replace(" ", "_")
            save_path = models_dir / f"{safe_name}.pkl"
            joblib.dump(clf, save_path)
            print(f"[train]   → saved to {save_path}")
            logger.info("Saved %s to %s", name, save_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to train %s: %s", name, exc)
            print(f"[train]   ✗ {name} failed: {exc}")

    print(f"[train] Trained {len(trained)}/{len(models)} models.")
    return trained
