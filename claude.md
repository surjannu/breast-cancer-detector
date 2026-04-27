# Breast Cancer Detector — Comprehensive Codebase Reference

> Generated: 2026-04-26. Intended as the single authoritative orientation document for new contributors.

---

## 1. Repository Overview

### Purpose
End-to-end binary classification system that predicts whether a breast tumour is **Benign** or **Malignant** using 30 numerical features derived from digitised fine needle aspirate (FNA) images.

### Problem Solved
Given clinical measurements from a biopsy image, the system automates the initial risk assessment that would otherwise require expert radiologist review. A high-recall model (minimising false negatives) is especially valuable here because missing a malignant case is far more costly than a false alarm.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        run_pipeline.py                          │
│  (one-command orchestrator — runs steps 1-6 sequentially)       │
└───────────────────┬─────────────────────────────────────────────┘
                    │
        ┌───────────▼───────────┐
        │   src/data/           │
        │   download.py         │  Step 1 – acquire raw CSV
        │   preprocess.py       │  Step 2 – clean / encode / scale / split
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   src/visualization/  │
        │   plots.py            │  Step 3 – EDA plots (class dist, heatmap)
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   src/models/         │
        │   train.py            │  Step 4 – fit 5 classifiers
        │   evaluate.py         │  Step 5 – score, rank, persist best model
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   src/visualization/  │
        │   plots.py            │  Step 6 – performance plots
        └───────────┬───────────┘
                    │
        ┌───────────▼───────────┐
        │   app.py              │
        │   Streamlit dashboard │  Interactive UI (4 pages)
        └───────────────────────┘
```

---

## 2. Project Structure

```
breast-cancer-detector/
│
├── run_pipeline.py             # ← ENTRY POINT: executes entire ML pipeline
├── app.py                      # ← ENTRY POINT: Streamlit web dashboard
├── requirements.txt            # Python dependencies
├── README.md                   # User-facing project overview
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── breast_cancer.csv       # 569 rows × 31 cols (downloaded/generated)
│   └── processed/
│       ├── breast_cancer_clean.csv # Unscaled clean data (used by EDA page)
│       ├── X_train.csv             # Scaled training features (455 × 30)
│       ├── X_test.csv              # Scaled test features   (114 × 30)
│       ├── y_train.csv             # Training labels  (455,)
│       ├── y_test.csv              # Test labels      (114,)
│       ├── feature_names.csv       # Ordered list of 30 feature names
│       └── scaler.pkl              # Fitted StandardScaler (joblib)
│
├── models/
│   ├── best_model.pkl              # Copy of the winning model
│   ├── best_model_name.txt         # Plain-text name ("Random Forest")
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl           # ← current best model
│   ├── svm.pkl
│   ├── xgboost.pkl
│   └── knn.pkl
│
├── reports/
│   ├── model_comparison.csv        # 5 models × 5 metrics table
│   └── figures/
│       ├── class_distribution.png
│       ├── correlation_heatmap.png
│       ├── feature_importance_best.png
│       ├── confusion_matrix_best.png
│       ├── roc_curves.png
│       └── model_comparison.png
│
└── src/                            # All importable business logic
    ├── __init__.py
    ├── data/
    │   ├── __init__.py             # Re-exports: download_data, preprocess_data
    │   ├── download.py
    │   └── preprocess.py
    ├── models/
    │   ├── __init__.py             # Re-exports: train_all_models, evaluate_models
    │   ├── train.py
    │   └── evaluate.py
    └── visualization/
        ├── __init__.py             # Re-exports: 6 plotting functions
        └── plots.py
```

**Key entry points:**

| Command | File | What it does |
|---|---|---|
| `python run_pipeline.py` | [run_pipeline.py](run_pipeline.py) | Runs the entire ML pipeline end-to-end |
| `streamlit run app.py` | [app.py](app.py) | Launches the interactive web dashboard |

---

## 3. Technology Stack

### Language
- **Python 3.9+**

### Machine Learning

| Library | Role |
|---|---|
| `scikit-learn` ≥ 1.1 | Logistic Regression, Random Forest, SVM, KNN, scaler, metrics |
| `xgboost` ≥ 1.7 | XGBoost classifier |
| `joblib` ≥ 1.2 | Model / scaler serialisation to `.pkl` |
| `ucimlrepo` ≥ 0.0.3 | Primary dataset download from UCI ML Repository |

### Data

| Library | Role |
|---|---|
| `pandas` ≥ 1.5 | DataFrames, CSV I/O |
| `numpy` ≥ 1.21 | Numerical arrays |

### Visualisation

| Library | Role |
|---|---|
| `matplotlib` ≥ 3.5 | Static PNG plots (pipeline steps 3 & 6) |
| `seaborn` ≥ 0.12 | Heatmaps, statistical styling |
| `plotly` ≥ 5.10 | Interactive charts inside Streamlit |
| `streamlit` ≥ 1.20 | Web dashboard framework |

### Optional / Future

| Library | Role |
|---|---|
| `shap` ≥ 0.41 | Model explainability (in requirements, not yet wired up) |
| `imbalanced-learn` ≥ 0.10 | Class imbalance handling (in requirements, not yet used) |

### Infrastructure
No cloud services, databases, or message queues. Everything is **local file-based**. Artefacts (CSVs, pickles, PNGs) are written to `data/`, `models/`, and `reports/`.

---

## 4. Core Logic & Data Flow

### Dataset
- **Source:** UCI Breast Cancer Wisconsin (Diagnostic) — id=17 via `ucimlrepo`, with sklearn fallback.
- **Size:** 569 samples, 30 numeric features, 1 binary label.
- **Classes:** Malignant (M → 1): 212 samples (37.3%), Benign (B → 0): 357 samples (62.7%).
- **Features:** 3 statistical groups (mean, standard error, worst) × 10 cellular measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension).

### End-to-End Data Flow

```
UCI ML Repo / sklearn (fallback)
        │
        ▼
breast_cancer.csv   (569 × 31, raw)
        │
        │  [preprocess.py]
        │  1. Drop unnamed index columns
        │  2. Fill NaN with column median (none present in current data)
        │  3. Drop duplicate rows
        │  4. Encode diagnosis: M→1, B→0  (sklearn convention is inverted — handled)
        │  5. StandardScaler.fit(X_train) → transform X_train & X_test
        │  6. Stratified 80/20 split (random_state=42)
        ▼
┌─────────────┬──────────────┐
│  X_train    │  y_train     │  455 samples, scaled
│  X_test     │  y_test      │  114 samples, scaled
│  scaler.pkl │ feature_names│
└─────────────┴──────────────┘
        │
        │  [train.py]
        │  Fit 5 classifiers on X_train / y_train
        ▼
logistic_regression.pkl  random_forest.pkl  svm.pkl  xgboost.pkl  knn.pkl
        │
        │  [evaluate.py]
        │  Score each model on X_test / y_test
        │  Select best by F1-Score
        ▼
best_model.pkl + best_model_name.txt + model_comparison.csv
        │
        │  [plots.py]
        │  Generate 6 PNG visualisations
        ▼
reports/figures/*.png
        │
        ▼
[app.py] — Streamlit loads all artefacts from disk
         → Overview | EDA | Model Performance | Predict
```

### Label Encoding Note
The UCI source uses string labels (`'M'`/`'B'`). The sklearn fallback uses integers with the *opposite* convention (0=Malignant, 1=Benign). `_encode_target()` in [src/data/preprocess.py](src/data/preprocess.py) handles both cases and normalises to **1=Malignant, 0=Benign** regardless of source.

---

## 5. Key Modules / Components

### [run_pipeline.py](run_pipeline.py)
Thin orchestrator. Calls `ensure_dirs()` then runs steps 1–6 in sequence, catching and re-raising errors with a clear step label. No ML logic lives here.

---

### [src/data/download.py](src/data/download.py)

| Function | Signature | Description |
|---|---|---|
| `download_data` | `(force: bool = False) → Path` | Public API. Tries UCI, falls back to sklearn, saves raw CSV. |
| `_download_from_ucimlrepo` | `() → pd.DataFrame` | Calls `ucimlrepo.fetch_ucirepo(id=17)`. |
| `_load_from_sklearn` | `() → pd.DataFrame` | Loads `sklearn.datasets.load_breast_cancer()`, inverts numeric labels. |

---

### [src/data/preprocess.py](src/data/preprocess.py)

| Function | Signature | Description |
|---|---|---|
| `preprocess_data` | `(raw_csv, test_size=0.2, random_state=42) → tuple[6]` | Full preprocessing pipeline. Returns `(X_train, X_test, y_train, y_test, feature_names, scaler)`. |
| `load_processed_data` | `() → tuple[6]` | Reloads artefacts from disk — used by the dashboard to avoid reprocessing on each page load. |
| `_encode_target` | `(series: pd.Series) → pd.Series` | Normalises string and numeric label formats to a consistent integer encoding. |

---

### [src/models/train.py](src/models/train.py)

| Function | Description |
|---|---|
| `build_models() → dict` | Returns `{name: unfitted_estimator}` for all 5 classifiers. |
| `train_all_models(X_train, y_train, models_dir) → dict` | Fits each classifier, saves `.pkl`, returns `{name: fitted_estimator}`. |

**Classifiers and key hyperparameters:**

| Model | Key Hyperparameters |
|---|---|
| Logistic Regression | `max_iter=1000`, `solver='lbfgs'` |
| Random Forest | `n_estimators=100`, `n_jobs=-1` |
| SVM | `kernel='rbf'`, `probability=True` |
| XGBoost | `n_estimators=100`, `eval_metric='logloss'`, `verbosity=0` |
| KNN | `n_neighbors=5`, `n_jobs=-1` |

All use `random_state=42` where applicable.

---

### [src/models/evaluate.py](src/models/evaluate.py)

| Function | Description |
|---|---|
| `evaluate_models(trained_models, X_test, y_test, selection_metric="F1-Score") → tuple[DataFrame, str]` | Scores all models, ranks by `selection_metric`, saves best model + comparison CSV. |
| `_compute_metrics(clf, X_test, y_test, name) → dict` | Returns 5-metric dict for one model. Handles `predict_proba`, `decision_function`, and plain `predict` fallbacks. |

**Metrics computed:** Accuracy, Precision, Recall, F1-Score, ROC-AUC (all rounded to 4 decimals).

---

### [src/visualization/plots.py](src/visualization/plots.py)

| Function | Output | Figure size |
|---|---|---|
| `plot_class_distribution(y)` | Bar + pie chart | 10×4 in |
| `plot_correlation_heatmap(df)` | Lower-triangle Pearson heatmap | 16×13 in |
| `plot_feature_importance(model, feature_names, top_n=15)` | Horizontal bar chart | 9×6 in |
| `plot_confusion_matrix(y_true, y_pred, model_name)` | 2×2 heatmap | 6×5 in |
| `plot_roc_curves(models, X_test, y_test)` | Overlaid ROC curves | 8×6 in |
| `plot_model_comparison(results_df)` | Grouped bar chart | 13×6 in |

All functions accept an optional `save_path`. When provided they save to PNG (DPI=150) and return the `plt.Figure` object. Colourblind-safe palette: Benign = `#0072B2`, Malignant = `#E69F00`.

---

### [app.py](app.py)

Multi-page Streamlit dashboard. Data is loaded once per session via `@st.cache_data` / `@st.cache_resource` decorators.

| Page | Key Features |
|---|---|
| **Overview** | Summary metric cards, interactive Plotly pie chart, best model bar chart |
| **EDA** | Per-feature histograms/box plots, interactive correlation heatmap, filtered descriptive stats |
| **Model Performance** | Highlighted metrics table, interactive ROC curves, confusion matrix with TP/FP/TN/FN cards |
| **Predict** | Manual slider input *or* CSV batch upload → scaled → predict → probability gauge chart |

A `pipeline_ready()` guard halts the app with a clear error message if `models/best_model.pkl` or `data/processed/X_test.csv` are missing.

---

## 6. Configuration & Environment

### No `.env` File
The project uses no environment variables. All paths are resolved relative to the project root via `pathlib.Path(__file__).resolve().parent` in each module.

### `requirements.txt`

```
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
xgboost>=1.7.0
joblib>=1.2.0
matplotlib>=3.5.0
seaborn>=0.12.0
plotly>=5.10.0
streamlit>=1.20.0
ucimlrepo>=0.0.3
shap>=0.41.0
imbalanced-learn>=0.10.0
```

### Setup

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
.venv\Scripts\activate          # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## 7. Build, Run, and Deployment

### Run the Pipeline

```bash
python run_pipeline.py
```

This single command executes all six steps sequentially and prints a final metrics summary to the console. Artefacts are written to `data/processed/`, `models/`, and `reports/`.

**Force re-download:** Delete `data/raw/breast_cancer.csv` before running, or pass `force=True` to `download_data()` in [run_pipeline.py](run_pipeline.py).

### Launch the Dashboard

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

The dashboard is read-only — it loads from disk and does not modify any pipeline artefacts.

### Run Tests
No test suite exists in the current codebase. See [Known Gaps](#11-known-gaps--improvements).

### Deployment
No deployment configuration is present. To deploy the Streamlit dashboard:

```bash
# Option A: Streamlit Community Cloud
# Push to GitHub, connect the repo at share.streamlit.io

# Option B: Docker (not currently configured)
# Would require a Dockerfile with: pip install -r requirements.txt && streamlit run app.py
```

---

## 8. Database / Data Models

No database is used. All persistence is local and file-based.

| Artefact | Format | Contents |
|---|---|---|
| `data/raw/breast_cancer.csv` | CSV | Raw download (569 × 31) |
| `data/processed/breast_cancer_clean.csv` | CSV | Cleaned, unscaled (569 × 31, encoded label) |
| `data/processed/X_train.csv` | CSV | Scaled training features (455 × 30) |
| `data/processed/X_test.csv` | CSV | Scaled test features (114 × 30) |
| `data/processed/y_train.csv` | CSV | Training labels (455 × 1) |
| `data/processed/y_test.csv` | CSV | Test labels (114 × 1) |
| `data/processed/feature_names.csv` | CSV | Ordered list of 30 feature names |
| `data/processed/scaler.pkl` | joblib | Fitted `StandardScaler` |
| `models/*.pkl` | joblib | Fitted classifier objects |
| `models/best_model_name.txt` | Plain text | Human-readable name of best model |
| `reports/model_comparison.csv` | CSV | 5 models × 5 metrics |

### Feature Schema

The 30 features follow the pattern `{statistic}_{measurement}`:

| Statistic group | Measurements (×10 each) |
|---|---|
| `_mean` | radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension |
| `_se` | same 10 (standard error) |
| `_worst` | same 10 (largest of 3 values per sample) |

---

## 9. External Integrations

| Integration | How Used |
|---|---|
| **UCI ML Repository** | `ucimlrepo.fetch_ucirepo(id=17)` — HTTP call at pipeline startup |
| **scikit-learn datasets** | Fallback when UCI is unavailable (`sklearn.datasets.load_breast_cancer()`) |
| **Streamlit** | Web UI framework; no external API calls are made by the dashboard itself |

No message queues, cloud storage, or third-party APIs are used.

---

## 10. Code Quality & Patterns

### Design Patterns

| Pattern | Where |
|---|---|
| **Registry / Strategy** | `build_models()` returns a dict of estimators — adding a new model requires one dict entry; the rest of the pipeline handles it automatically |
| **Facade** | `run_pipeline.py` hides all step complexity behind a single orchestrator |
| **Factory with fallback** | `download_data()` tries the primary source and falls back transparently |
| **Caching** | Streamlit `@st.cache_data` / `@st.cache_resource` prevents redundant disk I/O on rerenders |
| **Guard clause** | `pipeline_ready()` in `app.py` short-circuits rendering if required artefacts are absent |

### Conventions
- **Reproducibility:** `random_state=42` used consistently across all stochastic operations.
- **Stratified splitting:** `stratify=y` in `train_test_split` preserves class ratios in both splits.
- **Scaler discipline:** `StandardScaler` is fit only on training data and then applied to test data and live predictions — no data leakage.
- **Logging:** All `src/` modules use `logging` with a uniform format: `%(asctime)s [%(levelname)s] %(name)s — %(message)s`.
- **Path handling:** `pathlib.Path` throughout; no string concatenation for file paths.
- **Public API via `__init__.py`:** Each sub-package explicitly re-exports its public functions, keeping `run_pipeline.py` imports clean.

### Shared Utilities
`_save_and_return()` in [src/visualization/plots.py](src/visualization/plots.py) is the only shared helper — it handles optional save-to-disk and returns the figure, keeping all six plot functions DRY.

---

## 11. Known Gaps & Improvements

| Issue | Detail |
|---|---|
| **No test suite** | Zero unit or integration tests. Regressions in encoding or preprocessing logic would be silent. |
| **No hyperparameter tuning** | All models use manually chosen defaults. `GridSearchCV` or `Optuna` could meaningfully improve scores. |
| **`shap` and `imbalanced-learn` unused** | Both are declared in `requirements.txt` but never imported. Implement or remove them. |
| **Force re-download not exposed via CLI** | The `force` flag must be edited in source; there is no CLI argument. |
| **Minimal input validation on Predict page** | CSV upload only checks column presence — no range or type validation on uploaded values. |
| **Implicit tie-breaking** | When models share the same F1-Score, the winner depends on DataFrame sort order rather than an explicit rule. |
| **No artefact versioning** | Re-running the pipeline silently overwrites all models and reports; no experiment tracking (e.g., MLflow). |
| **No deployment configuration** | No Dockerfile, CI/CD pipeline, or cloud deployment config exists. |

### Suggested Improvements
1. Add `pytest` tests for `_encode_target`, `preprocess_data`, and `_compute_metrics`.
2. Add a `Makefile` with targets `make pipeline`, `make app`, and `make test`.
3. Wire up `shap` for per-prediction feature attribution on the Predict page.
4. Expose `--force-download` as a CLI argument via `argparse` in `run_pipeline.py`.
5. Add MLflow tracking so each pipeline run produces a logged experiment with metrics and artefact URIs.
6. Evaluate SMOTE (via `imbalanced-learn`) to address the 63/37 class imbalance before training.

---

## 12. Summary

- **What it is:** A self-contained, single-command breast cancer classification pipeline with an interactive Streamlit dashboard, built on the UCI Wisconsin Diagnostic dataset (569 samples, 30 features, binary label).
- **How to run:** `python run_pipeline.py` runs everything (download → preprocess → train → evaluate → visualise); `streamlit run app.py` opens the interactive UI.
- **Models:** Five classifiers are trained and compared; the winner by F1-Score (currently **Random Forest**, F1=0.963, Accuracy=0.974) is persisted as `models/best_model.pkl`.
- **Architecture:** Strictly layered — `src/data` → `src/models` → `src/visualization` — with no circular imports. All artefacts are plain files (CSV, pkl, PNG), making the pipeline easy to inspect and debug without a database.
- **Dashboard:** Four pages — project overview, exploratory data analysis, model comparison, and live inference (manual sliders or CSV batch upload) — all with interactive Plotly charts.
- **Reproducibility:** Fixed seeds, stratified splits, and a persisted scaler ensure identical results on every pipeline run.
- **Prediction pipeline:** User input → `StandardScaler.transform()` (using the *training-time* scaler) → `best_model.predict_proba()` → probability gauge chart. Scaler reuse is intentional and correct.
- **Extensibility:** Adding a new classifier requires only one entry in `build_models()`; serialisation, evaluation, and the dashboard ROC curve page handle it automatically.
- **Highest-priority gap:** No automated tests and no experiment tracking — these would be the highest-value next contributions before this project is used in any production or clinical context.
