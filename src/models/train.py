"""Train multiple classifiers on the breast cancer dataset."""

import logging
from pathlib import Path

import joblib
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
            n_jobs=1,  # single-threaded; GridSearchCV(n_jobs=-1) parallelises the search
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
            n_jobs=1,  # single-threaded; GridSearchCV(n_jobs=-1) parallelises the search
        ),
    }


def build_param_grids() -> dict:
    """Return hyperparameter search grids for GridSearchCV.

    Keys are prefixed with 'clf__' to target the classifier step inside the
    imblearn Pipeline (smote → clf).
    """
    return {
        "Logistic Regression": {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__solver": ["lbfgs", "liblinear"],
        },
        "Random Forest": {
            "clf__n_estimators": [100, 200, 300],
            "clf__max_depth": [None, 5, 10, 20],
            "clf__min_samples_split": [2, 5, 10],
        },
        "SVM": {
            "clf__C": [0.1, 1, 10, 100],
            "clf__gamma": ["scale", "auto"],
        },
        "XGBoost": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [3, 5, 7],
            "clf__learning_rate": [0.01, 0.1, 0.2],
        },
        "KNN": {
            "clf__n_neighbors": [3, 5, 7, 9, 11],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean", "manhattan"],
        },
    }


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    models_dir: Path | None = None,
) -> dict:
    """Tune hyperparameters via GridSearchCV with SMOTE inside each CV fold.

    SMOTE is placed inside an imblearn Pipeline so resampling happens
    independently within each fold's training split — synthetic samples never
    leak into the validation fold, avoiding inflated CV scores.

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

    models = build_models()
    param_grids = build_param_grids()
    trained: dict = {}

    for name, clf in models.items():
        print(f"[train] Tuning & training {name} …")
        logger.info("GridSearchCV tuning %s", name)
        try:
            # SMOTE runs only on each fold's training split inside GridSearchCV.
            pipeline = ImbPipeline([
                ("smote", SMOTE(random_state=42)),
                ("clf", clf),
            ])

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grids[name],
                scoring="f1",
                cv=5,
                n_jobs=-1,  # parallelise fold/param combos; estimators use n_jobs=1
                refit=True,
            )
            grid_search.fit(X_train, y_train)
            trained[name] = grid_search

            # Strip clf__ prefix for readable display
            display_params = {
                k.replace("clf__", ""): v
                for k, v in grid_search.best_params_.items()
            }
            print(f"[train]   → Best params : {display_params}")
            print(f"[train]   → Best CV F1  : {grid_search.best_score_:.4f}")
            logger.info(
                "Best params for %s: %s  (CV F1=%.4f)",
                name, display_params, grid_search.best_score_,
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
