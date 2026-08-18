"""
Breast Tumour Malignancy Screening - interactive model explorer
ML Assignment 2 | M.Tech (AIML/DSE), BITS Pilani WILP

Upload the held-out screening set (test_data.csv), pick a classifier and
inspect how it behaves: headline metrics, confusion matrix, classification
report, ROC curve and per-patient predictions.
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
    roc_curve,
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


def classify(proba: np.ndarray, cutoff: float) -> np.ndarray:
    """Turn malignancy probabilities into labels at a chosen cut-off.

    A strict `>` is deliberate. scikit-learn's own `.predict()` takes the
    argmax of the class probabilities, which sends an exact 50/50 tie to the
    *negative* class. Decision-tree leaves genuinely can hold p = 0.500, so
    using `>=` here would silently disagree with `.predict()` and this app
    would report different numbers from the training script. With `>`, a
    cut-off of 0.50 reproduces `.predict()` exactly.
    """
    return (proba > cutoff).astype(int)


def evaluate(model, X, y_true, cutoff: float = 0.50):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = classify(y_proba, cutoff)
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
    return scores, y_pred, y_proba


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


def plot_roc(curves: dict):
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for label, (fpr, tpr, auc_value) in curves.items():
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{label} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.9, label="Chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC on uploaded screening set", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")
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

    st.header("3 · Decision threshold")
    threshold = st.slider(
        "Malignant probability cut-off", 0.05, 0.95, 0.50, 0.05,
        help="Lower the cut-off to trade precision for recall — in screening, "
             "a missed malignancy costs more than a false alarm.",
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

    st.caption(
        f"Decision cut-off: **{threshold:.2f}** — at the default 0.50 this table "
        "reproduces the comparison table in the README exactly."
    )

    rows, curves = [], {}
    for label, model in models.items():
        if has_labels:
            scores, _, proba = evaluate(model, X, y, threshold)
            rows.append({"ML Model Name": label, **scores})
            fpr, tpr, _ = roc_curve(y, proba)
            curves[label] = (fpr, tpr, scores["AUC"])

    if has_labels:
        table = pd.DataFrame(rows).sort_values("MCC", ascending=False)
        st.dataframe(
            table.style.format({c: "{:.4f}" for c in table.columns[1:]})
                 .background_gradient(cmap="Greens", subset=list(table.columns[1:])),
            width="stretch", hide_index=True,
        )
        best = table.iloc[0]["ML Model Name"]
        st.info(f"🏆 Highest MCC on this data: **{best}**")

        left, right = st.columns([1.1, 1])
        with left:
            st.pyplot(plot_roc(curves))
        with right:
            melted = table.melt(id_vars="ML Model Name",
                                var_name="Metric", value_name="Score")
            fig, ax = plt.subplots(figsize=(5.4, 4.0))
            sns.barplot(melted, x="Metric", y="Score", hue="ML Model Name", ax=ax)
            ax.set_ylim(0.75, 1.005)
            ax.set_xlabel("")
            ax.legend(fontsize=6, ncol=2, loc="lower left")
            ax.tick_params(axis="x", labelrotation=30, labelsize=8)
            ax.set_title("Metric comparison", fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)
    else:
        preds = pd.DataFrame({
            label: np.where(classify(m.predict_proba(X)[:, 1], threshold) == 1,
                            "Malignant", "Benign")
            for label, m in models.items()
        })
        st.dataframe(preds, width="stretch")

else:
    model = models[model_choice]
    proba = model.predict_proba(X)[:, 1]
    pred = classify(proba, threshold)

    st.subheader(f"{model_choice} — threshold {threshold:.2f}")

    if has_labels:
        cols = st.columns(6)
        headline = {
            "Accuracy": accuracy_score(y, pred),
            "AUC": roc_auc_score(y, proba),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "MCC": matthews_corrcoef(y, pred),
        }
        for col, (name, value) in zip(cols, headline.items()):
            col.metric(name, f"{value:.4f}")

        tab_cm, tab_report, tab_roc = st.tabs(
            ["Confusion matrix", "Classification report", "ROC curve"]
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
                y, pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
            )
            st.dataframe(pd.DataFrame(report).T.round(4),
                         width="stretch")
        with tab_roc:
            fpr, tpr, _ = roc_curve(y, proba)
            st.pyplot(plot_roc({model_choice: (fpr, tpr, roc_auc_score(y, proba))}))

    st.subheader("Per-patient predictions")
    out = pd.DataFrame({
        "P(malignant)": proba.round(4),
        "Prediction": np.where(pred == 1, "Malignant", "Benign"),
    })
    if has_labels:
        out.insert(0, "Actual", np.where(y == 1, "Malignant", "Benign"))
        out["Correct"] = np.where(pred == y.values, "✓", "✗")
    st.dataframe(out, width="stretch", height=320)
    st.download_button(
        "⬇ Download predictions as CSV",
        out.to_csv(index=False).encode(),
        file_name=f"predictions_{model_choice.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )
