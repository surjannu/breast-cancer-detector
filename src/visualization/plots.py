"""Visualization functions for EDA and model performance."""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving to files

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_curve, auc

logger = logging.getLogger(__name__)

# ── Consistent colour palette ──────────────────────────────────────────────────
PALETTE = {"Benign": "#2ecc71", "Malignant": "#e74c3c"}
BACKGROUND = "#f8f9fa"


def _save_and_return(fig: plt.Figure, save_path: Optional[Path]) -> plt.Figure:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved figure to %s", save_path)
    return fig


# ── 1. Class distribution ──────────────────────────────────────────────────────
def plot_class_distribution(
    y: np.ndarray | pd.Series,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Bar + pie charts showing Benign vs Malignant distribution."""
    y = np.asarray(y)
    labels = ["Benign (0)", "Malignant (1)"]
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    colours = [PALETTE["Benign"], PALETTE["Malignant"]]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), facecolor=BACKGROUND)
    fig.suptitle("Class Distribution", fontsize=14, fontweight="bold")

    # Bar
    bars = axes[0].bar(labels, counts, color=colours, edgecolor="white", width=0.5)
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sample Counts")
    for bar, cnt in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, str(cnt),
                     ha="center", va="bottom", fontweight="bold")

    # Pie
    axes[1].pie(counts, labels=labels, colors=colours, autopct="%1.1f%%",
                startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[1].set_title("Proportion")

    fig.tight_layout()
    return _save_and_return(fig, save_path)


# ── 2. Correlation heatmap ─────────────────────────────────────────────────────
def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Correlation heatmap of all numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(16, 13), facecolor=BACKGROUND)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        cmap="RdYlGn",
        center=0,
        vmin=-1,
        vmax=1,
        annot=False,
        linewidths=0.4,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=15)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


# ── 3. Feature importance ──────────────────────────────────────────────────────
def plot_feature_importance(
    model,
    feature_names: list[str],
    save_path: Optional[Path] = None,
    top_n: int = 15,
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        title = f"Feature Importances — {type(model).__name__}"
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
        title = f"Feature Coefficients (|coef|) — {type(model).__name__}"
    else:
        logger.warning("Model %s has no feature_importances_ or coef_.", type(model).__name__)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Feature importance not available for this model.",
                ha="center", va="center", transform=ax.transAxes)
        return _save_and_return(fig, save_path)

    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    vals = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=BACKGROUND)
    colours = plt.cm.viridis(np.linspace(0.2, 0.85, len(vals)))
    bars = ax.barh(names[::-1], vals[::-1], color=colours[::-1], edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    fig.tight_layout()
    return _save_and_return(fig, save_path)


# ── 4. Confusion matrix ────────────────────────────────────────────────────────
def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Benign (0)", "Malignant (1)"])

    fig, ax = plt.subplots(figsize=(6, 5), facecolor=BACKGROUND)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save_and_return(fig, save_path)


# ── 5. ROC curves ─────────────────────────────────────────────────────────────
def plot_roc_curves(
    models: dict,
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Overlaid ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=BACKGROUND)
    cmap = plt.cm.tab10
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.50)")

    for idx, (name, clf) in enumerate(models.items()):
        if hasattr(clf, "predict_proba"):
            y_score = clf.predict_proba(X_test)[:, 1]
        elif hasattr(clf, "decision_function"):
            y_score = clf.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=cmap(idx / len(models)),
                label=f"{name} (AUC={roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_and_return(fig, save_path)


# ── 6. Model comparison bar chart ─────────────────────────────────────────────
def plot_model_comparison(
    results_df: pd.DataFrame,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Grouped bar chart comparing all models across metrics."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    models = results_df["Model"].tolist()
    x = np.arange(len(models))
    n_metrics = len(metrics)
    width = 0.15

    fig, ax = plt.subplots(figsize=(13, 6), facecolor=BACKGROUND)
    cmap = plt.cm.Set2

    for i, metric in enumerate(metrics):
        vals = results_df[metric].tolist()
        offset = (i - n_metrics / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=metric, color=cmap(i / n_metrics), edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return _save_and_return(fig, save_path)
