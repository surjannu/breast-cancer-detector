"""Evaluate trained models and select the best one."""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


def _compute_metrics(clf, X_test: np.ndarray, y_test: np.ndarray, name: str) -> dict:
    y_pred = clf.predict(X_test)

    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_prob = clf.decision_function(X_test)
    else:
        y_prob = y_pred.astype(float)

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
    }


def evaluate_models(
    trained_models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    selection_metric: str = "F1-Score",
) -> tuple[pd.DataFrame, str]:
    """Evaluate all models; save comparison CSV and best model.

    Args:
        trained_models: Dict of {name: fitted estimator}.
        X_test: Test features.
        y_test: True labels.
        selection_metric: Metric used to pick the best model.

    Returns:
        (results_df, best_model_name)
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for name, clf in trained_models.items():
        metrics = _compute_metrics(clf, X_test, y_test, name)
        records.append(metrics)
        logger.info("Evaluated %s: %s", name, metrics)

    results_df = pd.DataFrame(records).sort_values(selection_metric, ascending=False).reset_index(drop=True)

    # ---- Print comparison table ----
    print("\n" + "=" * 70)
    print(" MODEL COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print("=" * 70)

    # ---- Select best model ----
    best_row = results_df.iloc[0]
    best_model_name: str = best_row["Model"]
    best_clf = trained_models[best_model_name]

    print(f"\n[evaluate] Best model: {best_model_name}  ({selection_metric}={best_row[selection_metric]:.4f})")
    logger.info("Best model: %s  %s=%.4f", best_model_name, selection_metric, best_row[selection_metric])

    # ---- Persist best model ----
    best_path = MODELS_DIR / "best_model.pkl"
    joblib.dump(best_clf, best_path)
    # Also save the best model name so the app can display it
    (MODELS_DIR / "best_model_name.txt").write_text(best_model_name)
    print(f"[evaluate] Best model saved to {best_path}")

    # ---- Save comparison CSV ----
    csv_path = REPORTS_DIR / "model_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"[evaluate] Model comparison saved to {csv_path}")

    return results_df, best_model_name
