"""
Breast Tumour Malignancy Screening - interactive model explorer
ML Assignment 2 | M.Tech (AIML/DSE), BITS Pilani WILP

Upload the held-out screening set (test_data.csv), pick a classifier and
inspect its evaluation metrics, confusion matrix and classification report.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ------------------------------------------------------------------ config
TARGET_COLUMN = "diagnosis_malignant"
CLASS_NAMES = ["Benign", "Malignant"]
ARTEFACT_DIR = Path(__file__).parent / "model" / "artifacts"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest (Ensemble)": "random_forest_ensemble.joblib",
}

st.set_page_config(
    page_title="Malignancy Screening Explorer",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem;}
      div[data-testid="stMetric"] {
          background: #f4f7fb;
          border: 1px solid #dde5f0;
          border-radius: 10px;
          padding: 12px 8px 8px 14px;
      }
      div[data-testid="stMetricValue"] {font-size: 1.5rem;}
      .headline {
          font-size: 2.0rem; font-weight: 700; color: #123a63; margin-bottom: 0;
      }
      .sub {color: #5a6b7d; margin-top: 0.2rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner=False)
def load_models() -> dict:
    loaded = {}
    for label, filename in MODEL_FILES.items():
        path = ARTEFACT_DIR / filename
        if path.exists():
            loaded[label] = joblib.load(path)
    return loaded


@st.cache_data(show_spinner=False)
def load_bundled_testset() -> pd.DataFrame | None:
    path = Path(__file__).parent / "test_data.csv"
    return pd.read_csv(path) if path.exists() else None


def classify(proba: np.ndarray) -> np.ndarray:
    """Label a patient malignant when P(malignant) exceeds 0.50.

    A strict `>` is deliberate. scikit-learn's own `.predict()` takes the
    argmax of the class probabilities, which sends an exact 50/50 tie to the
    *negative* class. Decision-tree leaves genuinely can hold p = 0.500, so
    using `>=` here would silently disagree with `.predict()` and this app
    would report different numbers from the training script.
    """
    return (proba > 0.50).astype(int)


def evaluate(model, X, y_true):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = classify(y_proba)
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return scores, y_pred


def plot_confusion(cm: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    sns.heatmap(
        cm, annot=True, fmt="d", cbar=False, cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
        annot_kws={"size": 13},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ header
st.markdown('<p class="headline">🔬 Breast Tumour Malignancy Screening</p>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="sub">Five classifiers trained on the Breast Cancer Wisconsin '
    "(Diagnostic) dataset. Positive class = <b>Malignant</b>.</p>",
    unsafe_allow_html=True,
)

models = load_models()
if not models:
    st.error(
        f"No model artefacts found in `{ARTEFACT_DIR}`. "
        "Run `python model/train_models.py` first."
    )
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("1 · Screening data")
    uploaded = st.file_uploader(
        "Upload test data (CSV)",
        type="csv",
        help="Must contain the 30 WDBC feature columns. A "
             f"`{TARGET_COLUMN}` column enables scoring.",
    )
    use_bundled = st.checkbox("Use bundled test_data.csv", value=uploaded is None)

    st.header("2 · Model")
    model_choice = st.selectbox(
        "Classifier",
        ["Compare all models"] + list(models.keys()),
        index=0,
    )

    st.caption("Assignment 2 · M.Tech AIML/DSE · BITS Pilani WILP")

# ------------------------------------------------------------------ data in
data = None
source_note = ""
if uploaded is not None:
    data = pd.read_csv(uploaded)
    source_note = f"Uploaded file: **{uploaded.name}**"
elif use_bundled:
    data = load_bundled_testset()
    source_note = "Using the repository's bundled **test_data.csv**"

if data is None:
    st.info("⬅️ Upload a CSV in the sidebar (or tick *Use bundled test_data.csv*) "
            "to begin.")
    st.stop()

st.success(f"{source_note} — {data.shape[0]} patients × {data.shape[1]} columns")

with st.expander("Preview the screening data"):
    st.dataframe(data.head(12), width="stretch")

has_labels = TARGET_COLUMN in data.columns
if not has_labels:
    st.warning(
        f"No `{TARGET_COLUMN}` column found — predictions will be shown, "
        "but metrics cannot be computed."
    )

X = data.drop(columns=[TARGET_COLUMN]) if has_labels else data.copy()
y = data[TARGET_COLUMN].astype(int) if has_labels else None

# validate feature schema against the trained pipelines
expected_path = ARTEFACT_DIR / "feature_order.joblib"
if expected_path.exists():
    expected = list(joblib.load(expected_path))
    missing = [c for c in expected if c not in X.columns]
    if missing:
        st.error(f"Upload is missing {len(missing)} required feature(s): "
                 f"{', '.join(missing[:6])}…")
        st.stop()
    X = X[expected]

# ------------------------------------------------------------------ results
if model_choice == "Compare all models":
    st.subheader("Model comparison on this screening set")

    if has_labels:
        rows = []
        for label, model in models.items():
            scores, _ = evaluate(model, X, y)
            rows.append({"ML Model Name": label, **scores})

        table = pd.DataFrame(rows).sort_values("MCC", ascending=False)
        st.dataframe(
            table.style.format({c: "{:.4f}" for c in table.columns[1:]}),
            width="stretch", hide_index=True,
        )
        st.info(f"🏆 Highest MCC on this data: "
                f"**{table.iloc[0]['ML Model Name']}**")
        st.caption("Select a single model in the sidebar to see its confusion "
                   "matrix and classification report.")
    else:
        preds = pd.DataFrame({
            label: np.where(classify(m.predict_proba(X)[:, 1]) == 1,
                            "Malignant", "Benign")
            for label, m in models.items()
        })
        st.dataframe(preds, width="stretch")

else:
    model = models[model_choice]
    st.subheader(model_choice)

    if has_labels:
        scores, pred = evaluate(model, X, y)

        cols = st.columns(6)
        for col, (name, value) in zip(cols, scores.items()):
            col.metric(name, f"{value:.4f}")

        tab_cm, tab_report = st.tabs(
            ["Confusion matrix", "Classification report"]
        )
        with tab_cm:
            cm = confusion_matrix(y, pred)
            c1, c2 = st.columns([1, 1])
            with c1:
                st.pyplot(plot_confusion(cm, model_choice))
            with c2:
                tn, fp, fn, tp = cm.ravel()
                st.markdown(
                    f"""
                    | Outcome | Count | Meaning |
                    |---|---|---|
                    | True negative | {tn} | benign, correctly cleared |
                    | False positive | {fp} | benign flagged as malignant |
                    | **False negative** | **{fn}** | **malignant missed — costliest error** |
                    | True positive | {tp} | malignant correctly caught |
                    """
                )
        with tab_report:
            report = classification_report(
                y, pred, target_names=CLASS_NAMES, output_dict=True,
                zero_division=0,
            )
            st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")
    else:
        proba = model.predict_proba(X)[:, 1]
        st.dataframe(
            pd.DataFrame({
                "Prediction": np.where(classify(proba) == 1,
                                       "Malignant", "Benign"),
            }),
            width="stretch",
        )
