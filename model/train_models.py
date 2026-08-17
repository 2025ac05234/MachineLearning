"""
====================================================================
 Breast Tumour Malignancy Screening - Model Training Pipeline
====================================================================
 Machine Learning Assignment 2  |  M.Tech (AIML/DSE), BITS Pilani WILP

 Trains and tunes five classifiers on the Breast Cancer Wisconsin
 (Diagnostic) dataset, evaluates them on a held-out screening set,
 and writes the fitted pipelines + evaluation artefacts to disk.

 Run:  python model/train_models.py
====================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------
# Reproducibility + project paths
# --------------------------------------------------------------------
SEED = 17                       # fixed so every rerun reproduces the report
HOLDOUT_FRACTION = 0.25         # 25% of patients kept aside as screening set
CV_FOLDS = 5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTEFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTEFACT_DIR.mkdir(parents=True, exist_ok=True)

POSITIVE_LABEL = "Malignant"
NEGATIVE_LABEL = "Benign"
TARGET_COLUMN = "diagnosis_malignant"


# --------------------------------------------------------------------
# Step 1 - Load and reframe the dataset
# --------------------------------------------------------------------
def build_screening_frame() -> pd.DataFrame:
    """Load WDBC and re-encode the target so that malignant = 1.

    scikit-learn ships this dataset with 0 = malignant / 1 = benign.
    For a screening problem the clinically interesting event is a
    *malignant* tumour, so we invert the encoding. Every precision /
    recall / MCC number in this project therefore refers to catching
    malignant cases, which is the metric a radiologist actually cares
    about.
    """
    bunch = load_breast_cancer(as_frame=True)
    frame = bunch.frame.copy()

    frame[TARGET_COLUMN] = (frame.pop("target") == 0).astype(int)

    # tidy column names: "mean radius" -> "mean_radius"
    frame.columns = [c.strip().replace(" ", "_") for c in frame.columns]
    return frame


def describe_frame(frame: pd.DataFrame) -> dict:
    counts = frame[TARGET_COLUMN].value_counts().to_dict()
    return {
        "n_instances": int(frame.shape[0]),
        "n_features": int(frame.shape[1] - 1),
        "n_malignant": int(counts.get(1, 0)),
        "n_benign": int(counts.get(0, 0)),
        "malignant_rate": round(float(frame[TARGET_COLUMN].mean()), 4),
        "missing_values": int(frame.isna().sum().sum()),
        "duplicate_rows": int(frame.duplicated().sum()),
    }


# --------------------------------------------------------------------
# Step 2 - Candidate models and their search grids
# --------------------------------------------------------------------
def candidate_models() -> dict:
    """Each entry is (pipeline, hyper-parameter grid).

    Scaling is applied only where the algorithm is distance- or
    magnitude-sensitive. Trees and Gaussian NB are scale-invariant, so
    forcing a scaler on them would add noise to the pipeline without
    changing a single split point.
    """
    return {
        "Logistic Regression": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(max_iter=5000, solver="liblinear",
                                           random_state=SEED)),
            ]),
            {"clf__C": [0.05, 0.1, 0.5, 1.0, 5.0],
             "clf__penalty": ["l1", "l2"]},
        ),
        "Decision Tree": (
            Pipeline([
                ("clf", DecisionTreeClassifier(random_state=SEED)),
            ]),
            {"clf__criterion": ["gini", "entropy"],
             "clf__max_depth": [3, 4, 5, 7, None],
             "clf__min_samples_leaf": [1, 3, 5, 8]},
        ),
        "kNN": (
            Pipeline([
                ("scale", StandardScaler()),
                ("clf", KNeighborsClassifier()),
            ]),
            {"clf__n_neighbors": [3, 5, 7, 9, 11, 15],
             "clf__weights": ["uniform", "distance"],
             "clf__p": [1, 2]},
        ),
        "Naive Bayes": (
            Pipeline([
                ("clf", GaussianNB()),
            ]),
            # WDBC features are continuous real-valued measurements, so the
            # Gaussian variant is the correct member of the NB family here
            # (Multinomial NB assumes non-negative count data).
            {"clf__var_smoothing": np.logspace(-11, -6, 6)},
        ),
        "Random Forest (Ensemble)": (
            Pipeline([
                ("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1)),
            ]),
            {"clf__n_estimators": [200, 400],
             "clf__max_depth": [None, 6, 10],
             "clf__min_samples_leaf": [1, 2, 4],
             "clf__max_features": ["sqrt", 0.4]},
        ),
    }


# --------------------------------------------------------------------
# Step 3 - Evaluation
# --------------------------------------------------------------------
def score_model(fitted, X_eval, y_eval) -> dict:
    y_pred = fitted.predict(X_eval)
    y_proba = fitted.predict_proba(X_eval)[:, 1]

    return {
        "Accuracy": accuracy_score(y_eval, y_pred),
        "AUC": roc_auc_score(y_eval, y_proba),
        "Precision": precision_score(y_eval, y_pred, zero_division=0),
        "Recall": recall_score(y_eval, y_pred, zero_division=0),
        "F1": f1_score(y_eval, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_eval, y_pred),
    }


def main() -> None:
    print("=" * 68)
    print(" Breast Tumour Malignancy Screening - training run")
    print("=" * 68)

    frame = build_screening_frame()
    profile = describe_frame(frame)
    print("\nDataset profile:")
    for key, value in profile.items():
        print(f"  {key:<18}: {value}")

    features = frame.drop(columns=[TARGET_COLUMN])
    labels = frame[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        features, labels,
        test_size=HOLDOUT_FRACTION,
        stratify=labels,          # keeps the 37% malignant rate in both splits
        random_state=SEED,
    )
    print(f"\nTrain set : {X_train.shape[0]} patients "
          f"({int(y_train.sum())} malignant)")
    print(f"Screening : {X_test.shape[0]} patients "
          f"({int(y_test.sum())} malignant)")

    # persist the held-out screening set - this is the CSV uploaded to the app
    test_frame = X_test.copy()
    test_frame[TARGET_COLUMN] = y_test.values
    test_frame.to_csv(PROJECT_ROOT / "test_data.csv", index=False)
    print(f"\nWrote test_data.csv  ->  {test_frame.shape}")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    results, chosen_params = [], {}
    for name, (pipeline, grid) in candidate_models().items():
        print(f"\n--- Tuning {name} "
              f"({np.prod([len(v) for v in grid.values()])} configs) ---")
        search = GridSearchCV(pipeline, grid, scoring="roc_auc",
                              cv=cv, n_jobs=-1, refit=True)
        search.fit(X_train, y_train)

        best = search.best_estimator_
        params = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        chosen_params[name] = {k: (float(v) if isinstance(v, (np.floating,)) else v)
                               for k, v in params.items()}
        print(f"  best CV AUC : {search.best_score_:.4f}")
        print(f"  best params : {params}")

        metrics = score_model(best, X_test, y_test)
        metrics["Model"] = name
        results.append(metrics)
        print("  holdout     : " +
              "  ".join(f"{k}={metrics[k]:.4f}" for k in
                        ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]))

        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        joblib.dump(best, ARTEFACT_DIR / f"{slug}.joblib")

        tn, fp, fn, tp = confusion_matrix(y_test, best.predict(X_test)).ravel()
        results[-1]["_cm"] = [int(tn), int(fp), int(fn), int(tp)]

    scoreboard = pd.DataFrame(results)
    cms = {r["Model"]: r.pop("_cm") for r in results}
    scoreboard = scoreboard.drop(columns=["_cm"])
    scoreboard = scoreboard[["Model", "Accuracy", "AUC", "Precision",
                             "Recall", "F1", "MCC"]]
    scoreboard.to_csv(ARTEFACT_DIR / "metrics_summary.csv", index=False)

    joblib.dump(list(features.columns), ARTEFACT_DIR / "feature_order.joblib")

    meta = {
        "seed": SEED,
        "holdout_fraction": HOLDOUT_FRACTION,
        "target_column": TARGET_COLUMN,
        "positive_label": POSITIVE_LABEL,
        "negative_label": NEGATIVE_LABEL,
        "dataset_profile": profile,
        "best_params": chosen_params,
        "confusion_matrices_tn_fp_fn_tp": cms,
    }
    (ARTEFACT_DIR / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print("\n" + "=" * 68)
    print(scoreboard.round(4).to_string(index=False))
    print("=" * 68)
    winner = scoreboard.sort_values(["MCC", "AUC"], ascending=False).iloc[0]
    print(f"Best by MCC then AUC: {winner['Model']}")
    print(f"Artefacts written to {ARTEFACT_DIR}")


if __name__ == "__main__":
    main()
