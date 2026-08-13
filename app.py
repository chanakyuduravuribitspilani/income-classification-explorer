"""
Income Classification Explorer — Streamlit front-end for ML Assignment 2.

Upload a test CSV (UCI Adult format with an `income` column), pick one of the five
trained models, and instantly see its six evaluation metrics, a confusion matrix, and a
full classification report. A comparison view scores every model on the same test set.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MODELS_DIR = Path("models")
TARGET_COLUMN = "income"
POSITIVE_LABEL = ">50K"

MODEL_CATALOG = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest.pkl",
}

st.set_page_config(page_title="Income Classification Explorer",
                   page_icon="💼", layout="wide")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def load_pipeline(file_name: str):
    return joblib.load(MODELS_DIR / file_name)


def encode_target(series: pd.Series) -> np.ndarray:
    return (series.astype(str).str.strip() == POSITIVE_LABEL).astype(int).values


def score_model(pipeline, X, y_true) -> dict:
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# --------------------------------------------------------------------------- #
# Sidebar — data + model controls
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Controls")
st.sidebar.markdown("Upload **test data** only (CSV) — the free tier stays fast this way.")

uploaded = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
use_sample = st.sidebar.checkbox("Use bundled test_data.csv", value=True)

chosen_model = st.sidebar.selectbox("Choose a model", list(MODEL_CATALOG.keys()),
                                    index=len(MODEL_CATALOG) - 1)

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("💼 Income Classification Explorer")
st.caption("UCI Adult dataset · predicting whether annual income exceeds \\$50K · "
           "ML Assignment 2")

# --------------------------------------------------------------------------- #
# Resolve the data source
# --------------------------------------------------------------------------- #
if uploaded is not None:
    test_df = pd.read_csv(uploaded)
    source_note = "Loaded from your uploaded file."
elif use_sample and Path("test_data.csv").exists():
    test_df = pd.read_csv("test_data.csv")
    source_note = "Loaded from the bundled `test_data.csv`."
else:
    st.info("👈 Upload a test CSV or tick **Use bundled test_data.csv** to begin.")
    st.stop()

if TARGET_COLUMN not in test_df.columns:
    st.error(f"The CSV must contain a `{TARGET_COLUMN}` column with the true labels.")
    st.stop()

X_test = test_df.drop(columns=[TARGET_COLUMN])
y_test = encode_target(test_df[TARGET_COLUMN])

st.success(source_note + f"  ({len(test_df):,} rows)")
with st.expander("Preview the test data"):
    st.dataframe(test_df.head(20), use_container_width=True)

# --------------------------------------------------------------------------- #
# Single-model evaluation
# --------------------------------------------------------------------------- #
st.subheader(f"📊 Evaluation — {chosen_model}")

pipeline = load_pipeline(MODEL_CATALOG[chosen_model])
metrics = score_model(pipeline, X_test, y_test)

cols = st.columns(6)
for col, (name, value) in zip(cols, metrics.items()):
    col.metric(name, f"{value:.3f}")

left, right = st.columns([1, 1.2])

with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_test, pipeline.predict(X_test))
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.markdown("**Classification report**")
    report = classification_report(y_test, pipeline.predict(X_test),
                                   target_names=["<=50K", ">50K"], output_dict=True,
                                   zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

# --------------------------------------------------------------------------- #
# All-models comparison
# --------------------------------------------------------------------------- #
st.subheader("🏆 Compare all models on this test set")

if st.button("Run comparison across all 5 models"):
    rows = []
    for name, file_name in MODEL_CATALOG.items():
        rows.append({"ML Model Name": name,
                     **score_model(load_pipeline(file_name), X_test, y_test)})
    board = pd.DataFrame(rows).set_index("ML Model Name").round(4)
    st.dataframe(board.style.background_gradient(cmap="Greens"),
                 use_container_width=True)
    st.success(f"Overall winner by F1 score: **{board['F1'].idxmax()}**")

st.caption("Built for ML Assignment 2 · models trained in model_training.ipynb")
