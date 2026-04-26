# 🔬 Breast Cancer Detector

An end-to-end machine learning pipeline for classifying breast tumours as **Benign** or **Malignant** using the [UCI Breast Cancer Wisconsin Diagnostic dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic).

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** ⭐ | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9929 |
| XGBoost | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9940 |
| SVM | 0.9737 | 1.0000 | 0.9286 | 0.9630 | 0.9954 |
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9512 | 0.9960 |
| KNN | 0.9561 | 0.9744 | 0.9048 | 0.9383 | 0.9816 |

## Project Structure

```
breast-cancer-detector/
├── requirements.txt
├── run_pipeline.py          # Main script — runs entire pipeline
├── app.py                   # Streamlit dashboard
├── data/
│   ├── raw/                 # Downloaded raw data
│   └── processed/           # Cleaned, scaled, split data
├── models/                  # Saved trained models (.pkl)
├── reports/
│   └── figures/             # Generated plots (PNG)
└── src/
    ├── data/
    │   ├── download.py      # Download UCI data (ucimlrepo → sklearn fallback)
    │   └── preprocess.py    # Clean, encode, scale, split
    ├── models/
    │   ├── train.py         # Train 5 classifiers
    │   └── evaluate.py      # Evaluate and select best model
    └── visualization/
        └── plots.py         # EDA and model visualisation functions
```

## Dataset

- **569 samples**, 30 numeric features
- **Target**: 0 = Benign, 1 = Malignant  
- Features are mean, SE, and worst values of 10 cell-nucleus measurements (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension)

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline (download data → train → evaluate → generate plots)
python run_pipeline.py

# 3. Launch the interactive dashboard
streamlit run app.py
```

## Pipeline Steps

1. **Data Download** — tries `ucimlrepo` first, falls back to `sklearn`
2. **Preprocessing** — missing value handling, deduplication, target encoding, StandardScaler, 80/20 stratified split
3. **EDA Visualisations** — class distribution, correlation heatmap
4. **Model Training** — Logistic Regression, Random Forest, SVM (RBF), XGBoost, KNN
5. **Evaluation** — accuracy, precision, recall, F1, ROC-AUC; best model saved to `models/best_model.pkl`
6. **Performance Plots** — confusion matrix, ROC curves, model comparison bar chart, feature importances

## Streamlit Dashboard

Four pages accessible via the sidebar:

| Page | Contents |
|---|---|
| 🏠 Overview | Key stats, class distribution pie chart, best model summary |
| 📊 EDA | Interactive feature distributions, correlation heatmap, descriptive statistics |
| 🏆 Model Performance | Comparison table, ROC curves, confusion matrix |
| 🔮 Predict | Manual slider input (30 features) or CSV upload with probability gauge |

This project is a smart detector of breast cancer
