# Breast Tumour Malignancy Screening — Classifier Comparison

**Machine Learning Assignment 2 · M.Tech (AIML/DSE) · BITS Pilani WILP**

Five classification models trained on the Breast Cancer Wisconsin (Diagnostic)
dataset, compared on six evaluation metrics and served through an interactive
Streamlit application.

🔗 **Live app:** https://github.com/2025ac05234/MachineLearning.git
🔗 **Repository:** `<PASTE YOUR GITHUB REPO URL HERE>`

---

## a. Problem Statement

A fine-needle aspirate (FNA) of a breast mass is digitised, and image-analysis
software measures the geometry of each cell nucleus in the sample. The clinical
question is binary: **is the mass malignant or benign?**

The task is framed here as a *screening* problem rather than a generic
classification exercise, and that framing drives two decisions that run through
the whole project:

1. **Malignant is the positive class (label = 1).** scikit-learn distributes
   this dataset with `0 = malignant`, `1 = benign`; the encoding is inverted in
   `model/train_models.py` so that precision, recall, F1 and MCC all describe
   the ability to *catch cancer*, which is the quantity a clinician cares about.
2. **False negatives are the expensive error.** A benign mass wrongly flagged as
   malignant costs a follow-up biopsy. A malignant mass wrongly cleared can cost
   a life. Models are therefore ranked primarily by **MCC** (which uses all four
   confusion-matrix cells) and by **AUC**, with raw accuracy treated as the
   least informative of the six metrics.

The goal of the assignment is to determine which of five standard classifiers
best supports this decision on this data, and to expose the comparison through a
deployed web application where an evaluator can upload the held-out test set and
reproduce every number.

---

## b. Dataset Description

| Property | Value |
|---|---|
| Name | Breast Cancer Wisconsin (Diagnostic) — WDBC |
| Source | UCI Machine Learning Repository (ID 17); bundled with scikit-learn as `load_breast_cancer` |
| Original donors | Dr. William H. Wolberg, W. Nick Street, Olvi L. Mangasarian — University of Wisconsin |
| Task type | Binary classification |
| Instances | **569** (assignment minimum: 500 ✔) |
| Features | **30** numeric, real-valued (assignment minimum: 12 ✔) |
| Target | `diagnosis_malignant` — 1 = Malignant, 0 = Benign |
| Class balance | 212 malignant (37.3%) / 357 benign (62.7%) |
| Missing values | 0 |
| Duplicate rows | 0 |

### Feature structure

Ten physical measurements are computed for each cell nucleus in an image:

`radius`, `texture`, `perimeter`, `area`, `smoothness`, `compactness`,
`concavity`, `concave_points`, `symmetry`, `fractal_dimension`

Each measurement is then summarised three ways across the nuclei in the sample —
**mean**, **standard error (se)**, and **worst** (the mean of the three largest
values) — giving 10 × 3 = 30 columns such as `mean_radius`, `radius_error` and
`worst_concave_points`.

### Properties that matter for modelling

- **Strong multicollinearity by construction.** `mean_radius`,
  `mean_perimeter` and `mean_area` are three views of the same geometry and
  correlate above 0.98. This is the single most important structural fact about
  the dataset, and it explains two of the results below: it violates the
  conditional-independence assumption that Gaussian Naive Bayes depends on, and
  it rewards the L1-penalised logistic model, which prunes the redundancy.
- **Wildly different scales.** `mean_area` runs into the thousands while
  `mean_fractal_dimension` sits around 0.06. Distance- and magnitude-sensitive
  learners (kNN, logistic regression) are therefore wrapped in a
  `StandardScaler` inside their pipelines; trees and Gaussian NB are
  scale-invariant and deliberately left unscaled.
- **Nearly linearly separable after standardisation**, which foreshadows the
  headline result.

### Preprocessing and split

- Target re-encoded so malignant = 1.
- Column names normalised (`mean radius` → `mean_radius`).
- **Stratified 75 / 25 split**, `random_state=17` → 426 training patients
  (159 malignant) and **143 held-out screening patients (53 malignant)**;
  stratification preserves the 37% malignant rate in both halves.
- The 143-row held-out set is saved as **`test_data.csv`** and is the file
  uploaded to the Streamlit app. No model ever sees these rows during training.
- Hyperparameters chosen by `GridSearchCV` with **5-fold stratified CV on the
  training half only**, scored on ROC-AUC. All reported metrics come from the
  untouched screening set.

---

## c. GitHub Repository Link

https://github.com/2025ac05234/MachineLearning.git

```
project-folder/
│-- app.py                                   Streamlit application
│-- requirements.txt                         pinned dependencies
│-- README.md                                this file
│-- test_data.csv                            143-row held-out screening set
│-- model/
     │-- train_models.py                     end-to-end training pipeline
     │-- Breast_Cancer_Model_Development.ipynb   EDA + modelling notebook
     │-- artifacts/
          │-- logistic_regression.joblib
          │-- decision_tree.joblib
          │-- knn.joblib
          │-- naive_bayes.joblib
          │-- random_forest_ensemble.joblib
          │-- feature_order.joblib           schema used to validate uploads
          │-- metrics_summary.csv            the comparison table below
          │-- run_metadata.json              seed, best params, confusion matrices
```

### Reproducing the results

```bash
git clone <repo-url> && cd <repo>
pip install -r requirements.txt
python model/train_models.py     # retrains, rewrites artifacts/ and test_data.csv
streamlit run app.py             # opens the app at localhost:8501
```

The seed is fixed at 17 throughout, so a rerun reproduces every figure in this
README exactly.

---

## d. Models Used

All five models were trained on the same 426-row training split and evaluated on
the same 143-row held-out screening set.

| Model | Pipeline | Tuned hyperparameters (5-fold CV, ROC-AUC) |
|---|---|---|
| Logistic Regression | StandardScaler → LogisticRegression (liblinear) | `C=1.0`, `penalty='l1'` |
| Decision Tree | DecisionTreeClassifier | `criterion='gini'`, `max_depth=5`, `min_samples_leaf=8` |
| kNN | StandardScaler → KNeighborsClassifier | `n_neighbors=9`, `weights='distance'`, `p=2` |
| Naive Bayes | GaussianNB | `var_smoothing=1e-08` |
| Random Forest (Ensemble) | RandomForestClassifier | `n_estimators=200`, `max_depth=6`, `min_samples_leaf=2`, `max_features=0.4` |

**Gaussian, not Multinomial NB:** the 30 features are continuous real-valued
measurements. Multinomial NB assumes non-negative count data, so the Gaussian
member of the family is the correct choice here.

### Comparison Table — evaluation metrics on the held-out screening set

*(positive class = Malignant; n = 143 patients, 53 malignant)*

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | **0.9860** | **0.9979** | 0.9811 | **0.9811** | **0.9811** | **0.9700** |
| Decision Tree | 0.9161 | 0.9461 | 0.9362 | 0.8302 | 0.8800 | 0.8193 |
| kNN | 0.9790 | 0.9918 | **1.0000** | 0.9434 | 0.9709 | 0.9555 |
| Naive Bayes | 0.9231 | 0.9885 | 0.9375 | 0.8491 | 0.8911 | 0.8343 |
| Random Forest (Ensemble) | 0.9580 | 0.9931 | 0.9608 | 0.9245 | 0.9423 | 0.9098 |

### Confusion matrices (TN / FP / FN / TP)

| Model | TN | FP | FN — *missed malignancies* | TP |
|---|---|---|---|---|
| Logistic Regression | 89 | 1 | **1** | 52 |
| Decision Tree | 87 | 3 | **9** | 44 |
| kNN | 90 | 0 | **3** | 50 |
| Naive Bayes | 87 | 3 | **8** | 45 |
| Random Forest (Ensemble) | 88 | 2 | **4** | 49 |

---

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | **Best model on this dataset.** Top score on five of six metrics (Accuracy 0.9860, AUC 0.9979, Recall 0.9811, F1 0.9811, MCC 0.9700) and the lowest error count of all — a single false negative and a single false positive out of 143 patients. The reason is structural rather than lucky: after standardisation the two classes are very close to linearly separable, so a linear decision boundary is genuinely the right hypothesis class. CV selected an **L1 penalty**, which drives **14 of the 30 coefficients to exactly zero** and neutralises the mean/perimeter/area redundancy instead of being confused by it — it keeps one member of each collinear block and discards the rest. The surviving weights are led by `worst_area` (+4.39), `radius_error` (+2.31), `worst_texture` (+1.37) and `mean_concave_points` (+1.17) — clinically sensible drivers of malignancy, which makes the model both the most accurate *and* the most explainable of the five. |
| **Decision Tree** | **Weakest model, by a clear margin** (MCC 0.8193, AUC 0.9461). It misses **9 of 53 malignancies** — nine times the winner's false-negative count. Two things hurt it. First, a single tree makes axis-aligned splits on one feature at a time, so it cannot represent the smooth diagonal boundary the data actually has; it approximates it with a staircase and loses recall at every step. Second, CV had to prune it hard (`max_depth=5`, `min_samples_leaf=8`) to control the overfitting an unrestricted tree shows on 426 rows, and that pruning is exactly what costs sensitivity. Its low AUC also reflects coarse probability estimates: a depth-5 tree can only emit as many distinct scores as it has leaves, so its ranking of patients is inherently blunt. |
| **kNN** | **Strong runner-up** (MCC 0.9555, Accuracy 0.9790) and the only model with **perfect precision — zero false positives**: every patient it flagged as malignant genuinely was. The trade-off is 3 missed malignancies, which for a screening tool is the wrong direction to err. Distance-weighted voting over 9 neighbours works well because standardisation puts all 30 features on comparable footing and the malignant cases form a compact region of that space. Two practical caveats: kNN is the only model here that must carry all 426 training rows at inference time, and its performance is entirely contingent on the scaler — remove it and `mean_area` alone would dominate every distance computation. |
| **Naive Bayes** | **The most interesting split between metrics in the whole table:** AUC 0.9885 — third best, essentially level with the Random Forest — but MCC only 0.8343 and 8 missed malignancies. The gap tells a specific story. A high AUC means the model *ranks* patients by risk almost as well as the winners; the poor MCC means its probabilities are badly calibrated, so the default 0.50 cut-off lands in the wrong place. That miscalibration is the direct consequence of the conditional-independence assumption: with `mean_radius`, `mean_perimeter` and `mean_area` correlating above 0.98, the model multiplies what is effectively the same evidence three times and produces over-confident, saturated posteriors. NB is the model that would benefit most from threshold tuning — which is why the app exposes a threshold slider. |
| **Random Forest (Ensemble)** | **Solid third place** (MCC 0.9098, AUC 0.9931, 4 false negatives) and a comfortable improvement on the single Decision Tree it is built from — MCC rises from 0.8193 to 0.9098, a textbook demonstration of variance reduction through bagging and feature subsampling. Notably it still does **not** beat plain logistic regression. This is the honest lesson of the exercise: ensembles are not automatically superior. Their advantage is capturing non-linear interactions, and with only 426 training rows of near-linearly-separable data there is little non-linearity to capture, so the extra capacity buys nothing while the axis-aligned base learners retain the same handicap as the single tree. It is, however, the most robust option — no scaling required, insensitive to outliers, and its top feature importances (`worst_perimeter` 0.279, `worst_area` 0.173, `worst_radius` 0.148, `worst_concave_points` 0.107) independently corroborate the drivers the logistic model selected — two very different algorithms agreeing on the same signal. |
| **Overall Winner for your dataset?** | **🏆 Logistic Regression.** It wins on Accuracy, AUC, Recall, F1 and MCC, and ties or leads on total errors (2 of 143). It also wins on the criteria beyond the table: it is the cheapest model to train and serve, the smallest artefact on disk (3 KB versus 470 KB for the forest), the only one whose decision function can be read directly as log-odds contributions per feature, and the one that misses the fewest cancers — the error that actually matters in this problem. The result is a reminder that a well-regularised linear model remains a serious baseline, and that on well-engineered, near-separable features it can outperform a tuned ensemble. **Caveat:** the 143-patient screening set is small, so differences of 1–2 misclassifications between Logistic Regression, kNN and Random Forest sit within sampling noise. The reliable conclusion is the *grouping* — {Logistic Regression, kNN, Random Forest} clearly ahead of {Naive Bayes, Decision Tree} — rather than the precise ordering at the top. |

---

## e. Streamlit Application

**Live app:** `<PASTE YOUR STREAMLIT APP URL HERE>`

### Required features

| # | Requirement | Where it appears in the app |
|---|---|---|
| a | **Dataset upload option (CSV)** | Sidebar → *1 · Screening data* → file uploader. Accepts `test_data.csv` (143 rows). A checkbox loads the bundled copy if no file is uploaded. |
| b | **Model selection dropdown** | Sidebar → *2 · Model* → selectbox with all five classifiers plus a **Compare all models** mode. |
| c | **Display of evaluation metrics** | Six metric cards (Accuracy, AUC, Precision, Recall, F1, MCC) for the selected model; a colour-graded comparison table across all five in Compare mode. |
| d | **Confusion matrix / classification report** | Tabbed panel: *Confusion matrix* (annotated heatmap plus a TN/FP/FN/TP breakdown table), *Classification report* (per-class precision/recall/F1/support), *ROC curve*. |

### Additional features built beyond the requirement

- **Decision-threshold slider (0.05 – 0.95)** — lets an evaluator trade precision
  for recall interactively and see the false-negative count respond. This is the
  screening argument from the problem statement made tangible, and it is the
  clearest way to see Naive Bayes' calibration problem.
- **Compare-all mode** — overlaid ROC curves for the five models and a grouped
  metric bar chart on one screen.
- **Upload schema validation** — the app checks an uploaded CSV against the
  saved `feature_order.joblib` and names any missing columns instead of throwing
  a stack trace.
- **Label-free operation** — if the uploaded CSV has no `diagnosis_malignant`
  column the app still produces predictions, and explains that metrics are
  unavailable.
- **Per-patient prediction table** with malignancy probabilities, a correctness
  flag, and a CSV download button.

### How to reproduce the assignment results in the app

1. Open the live app link.
2. Leave *Use bundled test_data.csv* ticked (or upload `test_data.csv` from the repo).
3. Select **Compare all models** — the table reproduces the comparison table above.
4. Select any individual model to see its confusion matrix and classification report.

---

## f. BITS Virtual Lab Execution

The complete pipeline — data loading, training, evaluation and the Streamlit app
— was executed on the BITS Virtual Lab environment. The screenshot evidence is
included in the submitted PDF.

---

## Tech Stack

`Python` · `scikit-learn 1.7.2` · `pandas` · `NumPy` · `Matplotlib` · `Seaborn` ·
`joblib` · `Streamlit` · deployed on **Streamlit Community Cloud**

## References

1. Dua, D. and Graff, C. (2019). *UCI Machine Learning Repository* — Breast
   Cancer Wisconsin (Diagnostic) Data Set. University of California, Irvine.
   https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
2. Street, W.N., Wolberg, W.H., Mangasarian, O.L. (1993). *Nuclear feature
   extraction for breast tumor diagnosis.* IS&T/SPIE International Symposium on
   Electronic Imaging: Science and Technology, vol. 1905, pp. 861–870.
3. Chicco, D. and Jurman, G. (2020). *The advantages of the Matthews correlation
   coefficient (MCC) over F1 score and accuracy in binary classification
   evaluation.* BMC Genomics 21, 6.
4. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12,
   pp. 2825–2830.
