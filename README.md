# 💼 Income Classification Explorer — ML Assignment 2

An end-to-end machine learning project that predicts whether a person's annual income
exceeds \$50K, using the UCI *Adult (Census Income)* dataset. I train five classification
models, compare them across six evaluation metrics, and serve the results through an
interactive Streamlit web app.

---

## a. Problem statement

Given demographic and employment attributes of an individual (age, education, occupation,
weekly hours, capital gains, etc.), I predict a **binary outcome**: does the person earn
**more than \$50K a year (`>50K`)** or **not (`<=50K`)**? This is a classic supervised
binary classification problem with a mild class imbalance (~24% of people earn `>50K`).

## b. Dataset description

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository — *Adult / Census Income* |
| Link | https://archive.ics.uci.edu/dataset/2/adult |
| Instances | 32,561 |
| Features | 14 (6 numeric + 8 categorical) |
| Target | `income` → `>50K` (1) / `<=50K` (0) |
| Task | Binary classification |
| Class balance | 75.9% `<=50K`, 24.1% `>50K` |

**Features:** `age`, `workclass`, `fnlwgt`, `education`, `education_num`, `marital_status`,
`occupation`, `relationship`, `race`, `sex`, `capital_gain`, `capital_loss`,
`hours_per_week`, `native_country`.

The dataset satisfies the assignment requirements (≥ 12 features, ≥ 500 instances).
Missing values (marked `?`) are imputed; numeric features are scaled and categorical
features are one-hot encoded inside each model pipeline.

## c. GitHub repository link

> **Repository:** https://github.com/chanakyuduravuribitspilani/income-classification-explorer

The repository contains the complete source code, `requirements.txt`, this `README.md`,
the trained model pipelines, and the test data CSV.

## d. Models used

All five models were trained on the **same dataset** with an identical preprocessing
pipeline and an 80/20 stratified train/test split. The six metrics below are computed on
the held-out test set (6,513 rows).

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8558 | 0.9078 | 0.7406 | 0.6173 | 0.6734 | 0.5858 |
| Decision Tree | 0.8503 | 0.8896 | 0.7058 | 0.6486 | 0.6760 | 0.5797 |
| kNN | 0.8455 | 0.8909 | 0.7063 | 0.6135 | 0.6567 | 0.5599 |
| Naive Bayes | 0.5382 | 0.7495 | 0.3374 | **0.9528** | 0.4983 | 0.3294 |
| Random Forest (Ensemble) | **0.8672** | **0.9203** | **0.7966** | 0.6020 | **0.6858** | **0.6132** |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong, well-balanced baseline. Its high AUC (0.908) shows the classes are largely linearly separable, and it trains almost instantly. A very reliable second-best model. |
| Decision Tree | Competitive accuracy and the best recall among the tree/linear models, but a single tree is prone to variance; its AUC is the lowest of the strong models, hinting at slight overfitting. |
| kNN | Decent overall, yet the weakest of the "good" models. Distance-based voting struggles a little with the high-dimensional one-hot encoded space, giving it the lowest F1 and MCC of the four strong models. |
| Naive Bayes | Clear outlier. Its Gaussian independence assumption is violated by the one-hot categorical features, so it over-predicts `>50K` — huge recall (0.95) but poor precision (0.34) and the lowest accuracy and MCC. |
| Random Forest (Ensemble) | Best model overall — tops five of the six metrics (Accuracy, AUC, Precision, F1 and MCC); only Recall goes to Naive Bayes. Averaging many de-correlated trees controls the variance seen in the single Decision Tree and gives the most balanced, trustworthy predictions. |
| **Overall Winner** | **Random Forest (Ensemble)** — best in five of six metrics: Accuracy (0.867), AUC (0.920), Precision (0.797), F1 (0.686) and MCC (0.613); only Recall goes to Naive Bayes. |

## e. Streamlit web application

The deployed app lets a user explore the trained models interactively.

> **Live app:** https://chanakyuduravuribitspilani-income-classification-exp-app-bncdbu.streamlit.app/

**Features implemented:**
- 📁 **Test-data CSV upload** (or use the bundled `test_data.csv`).
- 🔽 **Model selection dropdown** for all five models.
- 📊 **Live display of all six evaluation metrics**.
- 🔵 **Confusion matrix and full classification report**, plus an always-visible
  comparison table that scores every model on the loaded test set and highlights the
  best value for each metric.

---

## 🗂️ Repository structure

```
.
├── app.py                       # Streamlit web application
├── requirements.txt             # Python dependencies (pinned)
├── raw_adult_income.csv         # Full raw dataset (from UCI)
├── test_data.csv                # Held-out test data for the app
├── README.md
├── notebooks/
│   ├── data_prep.py             # Shared loading, preprocessing & train/test split
│   ├── logistic_regression.ipynb
│   ├── decision_tree.ipynb
│   ├── knn.ipynb
│   ├── naive_bayes.ipynb
│   └── random_forest.ipynb
├── saved_models/                # One trained pipeline per model (.pkl)
└── metrics/                     # One metrics file per model (.csv)
```

## ▶️ Run locally

```bash
pip install -r requirements.txt
# (optional) retrain a model: open any notebook in notebooks/ and run all cells
streamlit run app.py
```

## 🚀 Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Click **New App**, select this repository and the `main` branch.
4. Set the main file to `app.py` and click **Deploy**.
