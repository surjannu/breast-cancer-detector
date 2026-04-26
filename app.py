"""Streamlit multi-page dashboard for the Breast Cancer Detector project."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    accuracy_score,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .main-header {font-size:2.2rem; font-weight:700; color:#1a1a2e; margin-bottom:0.2rem;}
    .sub-header  {font-size:1rem;   color:#555;      margin-bottom:1.5rem;}
    .metric-card {
        background:#f0f4ff; border-radius:10px; padding:1rem 1.5rem;
        box-shadow:0 2px 6px rgba(0,0,0,.08); text-align:center;
    }
    .metric-card h3 {margin:0; font-size:2rem; color:#1a1a2e;}
    .metric-card p  {margin:0; color:#666; font-size:0.9rem;}
    .benign    {color:#0072B2; font-weight:700; font-size:1.4rem;}
    .malignant {color:#E69F00; font-weight:700; font-size:1.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Data / model loaders (cached) ─────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_processed():
    """Load preprocessed data from disk."""
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv").values
    X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv").values
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").values.ravel()
    y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()
    features = pd.read_csv(PROCESSED_DIR / "feature_names.csv")["feature"].tolist()
    return X_train, X_test, y_train, y_test, features


@st.cache_data(show_spinner=False)
def load_clean_df():
    return pd.read_csv(PROCESSED_DIR / "breast_cancer_clean.csv")


@st.cache_resource(show_spinner=False)
def load_best_model():
    return joblib.load(MODELS_DIR / "best_model.pkl")


@st.cache_data(show_spinner=False)
def load_comparison():
    return pd.read_csv(REPORTS_DIR / "model_comparison.csv")


def pipeline_ready() -> bool:
    return (MODELS_DIR / "best_model.pkl").exists() and (PROCESSED_DIR / "X_test.csv").exists()


def best_model_name() -> str:
    name_file = MODELS_DIR / "best_model_name.txt"
    if name_file.exists():
        return name_file.read_text().strip()
    return "Best Model"


# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.markdown("## 🔬 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Overview", "📊 EDA", "🏆 Model Performance", "🔮 Predict"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Breast Cancer Detector · ML Pipeline")

# ── Guard: pipeline must have run first ───────────────────────────────────────
if not pipeline_ready():
    st.markdown('<p class="main-header">🔬 Breast Cancer Detector</p>', unsafe_allow_html=True)
    st.error(
        "**Pipeline output not found.**\n\n"
        "Please run the full pipeline first:\n"
        "```bash\npython run_pipeline.py\n```\n"
        "This will download the data, train models, and generate all artefacts.",
        icon="⚠️",
    )
    st.stop()

# ── Load shared data ───────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test, feature_names = load_processed()
df_clean = load_clean_df()
best_clf = load_best_model()
model_name = best_model_name()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<p class="main-header">🔬 Breast Cancer Detector</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Machine-learning pipeline for diagnosing breast tumours '
        'as <b>Benign</b> or <b>Malignant</b> using the UCI Wisconsin Diagnostic dataset.</p>',
        unsafe_allow_html=True,
    )

    # Key stats
    n_samples  = len(df_clean)
    n_features = len(feature_names)
    n_benign   = int((df_clean["diagnosis"] == 0).sum())
    n_malignant = int((df_clean["diagnosis"] == 1).sum())

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, color in [
        (c1, "Total Samples",    n_samples,   "#3498db"),
        (c2, "Features",         n_features,  "#9b59b6"),
        (c3, "Benign (0)",       n_benign,    "#0072B2"),
        (c4, "Malignant (1)",    n_malignant, "#E69F00"),
    ]:
        col.markdown(
            f'<div class="metric-card"><h3 style="color:{color};">{value}</h3>'
            f'<p>{label}</p></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("Class Distribution")
        fig = px.pie(
            names=["Benign (0)", "Malignant (1)"],
            values=[n_benign, n_malignant],
            color_discrete_sequence=["#0072B2", "#E69F00"],
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(margin=dict(t=20, b=20, l=20, r=20), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Best Model Summary")
        if (REPORTS_DIR / "model_comparison.csv").exists():
            comp = load_comparison()
            best_row = comp[comp["Model"] == model_name].iloc[0]
            metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
            fig2 = go.Figure(go.Bar(
                x=[best_row[m] for m in metrics],
                y=metrics,
                orientation="h",
                marker_color=["#3498db", "#9b59b6", "#e67e22", "#0072B2", "#E69F00"],
                text=[f"{best_row[m]:.3f}" for m in metrics],
                textposition="outside",
            ))
            fig2.update_layout(
                xaxis_range=[0, 1.1],
                xaxis_title="Score",
                title=f"{model_name}",
                margin=dict(t=40, b=20, l=10, r=30),
                height=280,
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("About the Dataset")
    st.markdown(
        """
The **UCI Breast Cancer Wisconsin (Diagnostic)** dataset contains 569 samples with 30
numeric features computed from digitised fine needle aspirate (FNA) images of breast
masses.  Ten real-valued features are computed for each cell nucleus:

> *radius, texture, perimeter, area, smoothness, compactness, concavity,
> concave points, symmetry, fractal dimension*

For each feature the **mean**, **standard error**, and **worst** (largest) value
across all cells are recorded, yielding 30 features total.

| Label | Meaning | Count |
|-------|---------|-------|
| 0 | Benign  | {b} |
| 1 | Malignant | {m} |
        """.format(b=n_benign, m=n_malignant)
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<p class="main-header">📊 Exploratory Data Analysis</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Feature Distributions", "Correlation Heatmap", "Statistics"])

    with tab1:
        st.subheader("Feature Distribution by Class")
        selected = st.selectbox("Select feature", feature_names)
        benign_vals    = df_clean[df_clean["diagnosis"] == 0][selected]
        malignant_vals = df_clean[df_clean["diagnosis"] == 1][selected]

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=benign_vals,    name="Benign (0)",    opacity=0.7,
                                    marker_color="#0072B2", nbinsx=30))
        fig.add_trace(go.Histogram(x=malignant_vals, name="Malignant (1)", opacity=0.7,
                                    marker_color="#E69F00", nbinsx=30))
        fig.update_layout(barmode="overlay", title=f"Distribution of {selected}",
                          xaxis_title=selected, yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)

        # Box plot
        fig2 = go.Figure()
        fig2.add_trace(go.Box(y=benign_vals,    name="Benign (0)",    marker_color="#0072B2"))
        fig2.add_trace(go.Box(y=malignant_vals, name="Malignant (1)", marker_color="#E69F00"))
        fig2.update_layout(title=f"Box Plot — {selected}", yaxis_title=selected)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.subheader("Feature Correlation Heatmap")
        feat_df = df_clean.drop(columns=["diagnosis"], errors="ignore")
        corr = feat_df.corr().round(2)

        fig = px.imshow(
            corr,
            color_continuous_scale="RdYlGn",
            zmin=-1, zmax=1,
            aspect="auto",
            text_auto=False,
        )
        fig.update_layout(height=700, margin=dict(t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        # Top correlated pairs
        st.subheader("Highly Correlated Pairs (|r| > 0.9)")
        pairs = (
            corr.where(np.tril(np.ones(corr.shape), k=-1).astype(bool))
                .stack()
                .reset_index()
        )
        pairs.columns = ["Feature A", "Feature B", "Correlation"]
        pairs = pairs[pairs["Correlation"].abs() > 0.9].sort_values("Correlation", ascending=False)
        st.dataframe(pairs.reset_index(drop=True), use_container_width=True)

    with tab3:
        st.subheader("Descriptive Statistics")
        group = st.radio("Show stats for", ["All", "Benign (0)", "Malignant (1)"], horizontal=True)
        subset = df_clean.drop(columns=["diagnosis"], errors="ignore")
        if group == "Benign (0)":
            subset = df_clean[df_clean["diagnosis"] == 0].drop(columns=["diagnosis"], errors="ignore")
        elif group == "Malignant (1)":
            subset = df_clean[df_clean["diagnosis"] == 1].drop(columns=["diagnosis"], errors="ignore")
        st.dataframe(subset.describe().T.round(4), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Performance":
    st.markdown('<p class="main-header">🏆 Model Performance</p>', unsafe_allow_html=True)

    comp = load_comparison()

    tab1, tab2, tab3 = st.tabs(["Comparison Table", "ROC Curves", "Confusion Matrix"])

    with tab1:
        st.subheader("Model Comparison")

        def highlight_best(s):
            is_best = s == s.max()
            return ["background-color:#d4edda; font-weight:bold" if v else "" for v in is_best]

        numeric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        styled = (
            comp.style
                .apply(highlight_best, subset=numeric_cols)
                .format({c: "{:.4f}" for c in numeric_cols})
        )
        st.dataframe(styled, use_container_width=True)
        st.caption(f"✅ Best model selected: **{model_name}** (highest F1-Score)")

        # Grouped bar chart
        comp_melt = comp.melt(id_vars="Model", value_vars=numeric_cols,
                               var_name="Metric", value_name="Score")
        fig = px.bar(
            comp_melt, x="Model", y="Score", color="Metric", barmode="group",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="All Models — All Metrics",
        )
        fig.update_layout(yaxis_range=[0, 1.08], xaxis_tickangle=-15)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("ROC Curves")

        # Load all saved models
        model_files = {
            "Logistic Regression": MODELS_DIR / "logistic_regression.pkl",
            "Random Forest":       MODELS_DIR / "random_forest.pkl",
            "SVM":                 MODELS_DIR / "svm.pkl",
            "XGBoost":             MODELS_DIR / "xgboost.pkl",
            "KNN":                 MODELS_DIR / "knn.pkl",
        }

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(dash="dash", color="grey"),
                                  name="Random (AUC=0.50)"))

        colours = px.colors.qualitative.Plotly
        for idx, (name, pkl_path) in enumerate(model_files.items()):
            if not pkl_path.exists():
                continue
            try:
                clf = joblib.load(pkl_path)
                if hasattr(clf, "predict_proba"):
                    y_score = clf.predict_proba(X_test)[:, 1]
                elif hasattr(clf, "decision_function"):
                    y_score = clf.decision_function(X_test)
                else:
                    continue
                fpr, tpr, _ = roc_curve(y_test, y_score)
                roc_auc = float(np.trapz(tpr, fpr))  # same as sklearn auc
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr, mode="lines",
                    line=dict(width=2.5, color=colours[idx % len(colours)]),
                    name=f"{name} (AUC={roc_auc:.3f})",
                ))
            except Exception:
                pass

        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            title="ROC Curves — All Models",
            legend=dict(x=0.6, y=0.1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Confusion Matrix — Best Model")
        y_pred = best_clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Benign (0)", "Malignant (1)"],
            y=["Benign (0)", "Malignant (1)"],
            text_auto=True,
            color_continuous_scale="Blues",
        )
        fig.update_layout(title=f"Confusion Matrix — {model_name}", height=400)
        st.plotly_chart(fig, use_container_width=True)

        tn, fp, fn, tp = cm.ravel()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("True Negative",  tn,  help="Correctly predicted Benign")
        c2.metric("False Positive", fp,  help="Benign predicted as Malignant")
        c3.metric("False Negative", fn,  help="Malignant predicted as Benign", delta=f"-{fn}" if fn else None)
        c4.metric("True Positive",  tp,  help="Correctly predicted Malignant")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown('<p class="main-header">🔮 Predict Diagnosis</p>', unsafe_allow_html=True)
    st.markdown(
        f'Using **{model_name}** to predict whether a tumour is '
        '<span class="benign">Benign</span> or <span class="malignant">Malignant</span>.',
        unsafe_allow_html=True,
    )

    mode = st.radio("Input mode", ["Manual (sliders)", "Upload CSV"], horizontal=True)
    st.markdown("---")

    # ── Feature statistics for slider ranges ──────────────────────────────────
    feat_stats = df_clean[feature_names].describe()

    def predict_and_display(input_array: np.ndarray, label: str = "") -> None:
        """Run model prediction and render result."""
        proba = None
        if hasattr(best_clf, "predict_proba"):
            proba = best_clf.predict_proba(input_array)[0]
            pred = int(np.argmax(proba))
            malignant_prob = float(proba[1])
        else:
            pred = int(best_clf.predict(input_array)[0])
            malignant_prob = float(pred)

        if label:
            st.write(f"**Sample:** {label}")

        if pred == 0:
            st.markdown('<p class="benign">✅ Prediction: BENIGN (0)</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="malignant">⚠️ Prediction: MALIGNANT (1)</p>', unsafe_allow_html=True)

        if proba is not None:
            col_b, col_m = st.columns(2)
            col_b.metric("Benign probability",    f"{proba[0]*100:.1f}%")
            col_m.metric("Malignant probability", f"{proba[1]*100:.1f}%")

            # Confidence gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=malignant_prob * 100,
                number={"suffix": "%", "font": {"size": 28}},
                title={"text": "Malignant Probability", "font": {"size": 16}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar": {"color": "#E69F00" if malignant_prob > 0.5 else "#0072B2"},
                    "steps": [
                        {"range": [0,  50], "color": "#d4edda"},
                        {"range": [50, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": 50,
                    },
                },
            ))
            fig.update_layout(height=280, margin=dict(t=30, b=10, l=20, r=20))
            st.plotly_chart(fig, use_container_width=True)

    # ── Manual input ──────────────────────────────────────────────────────────
    if mode == "Manual (sliders)":
        st.subheader("Enter feature values")

        # Display 3 sliders per row
        input_vals: dict[str, float] = {}
        cols_per_row = 3
        feature_chunks = [feature_names[i:i+cols_per_row]
                          for i in range(0, len(feature_names), cols_per_row)]

        for chunk in feature_chunks:
            cols = st.columns(cols_per_row)
            for col, feat in zip(cols, chunk):
                fmin = float(feat_stats.loc["min", feat])
                fmax = float(feat_stats.loc["max", feat])
                fmean = float(feat_stats.loc["mean", feat])
                input_vals[feat] = col.number_input(
                    feat,
                    min_value=round(fmin, 6),
                    max_value=round(fmax, 6),
                    value=round(fmean, 6),
                    format="%.5f",
                    key=feat,
                )

        if st.button("🔮 Predict", type="primary"):
            scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")
            raw_input = np.array([[input_vals[f] for f in feature_names]])
            scaled_input = scaler.transform(raw_input)
            st.markdown("---")
            predict_and_display(scaled_input)

    # ── CSV upload ─────────────────────────────────────────────────────────────
    else:
        st.subheader("Upload a CSV file")
        st.caption(
            "The CSV must contain the 30 feature columns "
            "(matching the feature names used during training). "
            "A 'diagnosis' column is optional (used for ground-truth comparison)."
        )

        uploaded = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded is not None:
            try:
                df_upload = pd.read_csv(uploaded)
                st.write(f"Loaded {len(df_upload)} row(s).")
                st.dataframe(df_upload.head(), use_container_width=True)

                # Align columns to expected feature order
                missing_feats = [f for f in feature_names if f not in df_upload.columns]
                if missing_feats:
                    st.error(f"Missing columns: {missing_feats}")
                    st.stop()

                X_upload = df_upload[feature_names].values
                scaler = joblib.load(PROCESSED_DIR / "scaler.pkl")
                X_upload_scaled = scaler.transform(X_upload)

                if st.button("🔮 Predict All", type="primary"):
                    predictions = []
                    proba_list  = []
                    for i, row in enumerate(X_upload_scaled):
                        sample = row.reshape(1, -1)
                        p = int(best_clf.predict(sample)[0])
                        if hasattr(best_clf, "predict_proba"):
                            prob = best_clf.predict_proba(sample)[0]
                        else:
                            prob = np.array([1 - p, p], dtype=float)
                        predictions.append(p)
                        proba_list.append(prob)

                    results = df_upload.copy()
                    results["Predicted"] = predictions
                    results["Predicted_Label"] = ["Malignant" if p == 1 else "Benign"
                                                  for p in predictions]
                    results["Benign_Prob"]    = [round(pr[0] * 100, 2) for pr in proba_list]
                    results["Malignant_Prob"] = [round(pr[1] * 100, 2) for pr in proba_list]

                    st.success(f"Predictions complete for {len(results)} sample(s).")
                    st.dataframe(
                        results[["Predicted_Label", "Benign_Prob", "Malignant_Prob"]],
                        use_container_width=True,
                    )

                    # Ground truth comparison
                    if "diagnosis" in df_upload.columns:
                        y_true = df_upload["diagnosis"].values
                        acc = accuracy_score(y_true, predictions)
                        st.metric("Accuracy on uploaded data", f"{acc:.4f}")

                    csv_out = results.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download predictions CSV",
                        data=csv_out,
                        file_name="predictions.csv",
                        mime="text/csv",
                    )

            except Exception as exc:
                st.error(f"Error processing file: {exc}")
