#!/usr/bin/env python3
"""Main pipeline script — run from project root with: python run_pipeline.py"""

import logging
import sys
from pathlib import Path

# ── Ensure project root is on the Python path ─────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def ensure_dirs() -> None:
    dirs = [
        "data/raw",
        "data/processed",
        "models",
        "reports/figures",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def main() -> None:
    print("\n" + "=" * 65)
    print("  BREAST CANCER DETECTION — ML PIPELINE")
    print("=" * 65 + "\n")

    # ── 0. Directory structure ─────────────────────────────────────────────────
    ensure_dirs()

    # ── 1. Download data ───────────────────────────────────────────────────────
    print("── STEP 1: Data Download ──────────────────────────────────────")
    from src.data.download import download_data
    raw_csv = download_data()

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    print("\n── STEP 2: Preprocessing ──────────────────────────────────────")
    from src.data.preprocess import preprocess_data
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess_data(raw_csv)

    # ── 3. EDA plots ──────────────────────────────────────────────────────────
    print("\n── STEP 3: EDA Visualisations ─────────────────────────────────")
    import pandas as pd
    import numpy as np
    from src.visualization.plots import (
        plot_class_distribution,
        plot_correlation_heatmap,
        plot_feature_importance,
        plot_confusion_matrix,
        plot_roc_curves,
        plot_model_comparison,
    )

    figures_dir = PROJECT_ROOT / "reports" / "figures"
    clean_csv = PROJECT_ROOT / "data" / "processed" / "breast_cancer_clean.csv"
    df_clean = pd.read_csv(clean_csv)
    y_all = df_clean["diagnosis"].values

    plot_class_distribution(y_all, save_path=figures_dir / "class_distribution.png")
    print("[pipeline] class_distribution.png saved.")

    plot_correlation_heatmap(
        df_clean.drop(columns=["diagnosis"], errors="ignore"),
        save_path=figures_dir / "correlation_heatmap.png",
    )
    print("[pipeline] correlation_heatmap.png saved.")

    # ── 4. Train models ────────────────────────────────────────────────────────
    print("\n── STEP 4: Model Training ─────────────────────────────────────")
    from src.models.train import train_all_models
    trained_models = train_all_models(X_train, y_train)

    # ── 5. Evaluate & select best ─────────────────────────────────────────────
    print("\n── STEP 5: Model Evaluation ───────────────────────────────────")
    from src.models.evaluate import evaluate_models
    results_df, best_name = evaluate_models(trained_models, X_test, y_test)

    # ── 6. Performance plots ───────────────────────────────────────────────────
    print("\n── STEP 6: Performance Visualisations ─────────────────────────")
    best_clf = trained_models[best_name]
    y_pred_best = best_clf.predict(X_test)

    plot_confusion_matrix(
        y_test, y_pred_best,
        model_name=best_name,
        save_path=figures_dir / "confusion_matrix_best.png",
    )
    print("[pipeline] confusion_matrix_best.png saved.")

    plot_roc_curves(
        trained_models, X_test, y_test,
        save_path=figures_dir / "roc_curves.png",
    )
    print("[pipeline] roc_curves.png saved.")

    plot_model_comparison(results_df, save_path=figures_dir / "model_comparison.png")
    print("[pipeline] model_comparison.png saved.")

    plot_feature_importance(
        best_clf, feature_names,
        save_path=figures_dir / "feature_importance_best.png",
    )
    print("[pipeline] feature_importance_best.png saved.")

    # ── 7. Summary ────────────────────────────────────────────────────────────
    best_metrics = results_df[results_df["Model"] == best_name].iloc[0]
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"  Best model  : {best_name}")
    print(f"  Accuracy    : {best_metrics['Accuracy']:.4f}")
    print(f"  Precision   : {best_metrics['Precision']:.4f}")
    print(f"  Recall      : {best_metrics['Recall']:.4f}")
    print(f"  F1-Score    : {best_metrics['F1-Score']:.4f}")
    print(f"  ROC-AUC     : {best_metrics['ROC-AUC']:.4f}")
    print("=" * 65)
    print("\nTo launch the interactive dashboard run:")
    print("  streamlit run app.py\n")


if __name__ == "__main__":
    main()
