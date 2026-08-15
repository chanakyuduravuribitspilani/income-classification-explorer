"""
Shared data preparation for the income-classification models.

Every per-model notebook imports this module so the dataset is loaded, cleaned and
split in exactly the same way. Keeping this logic in one place guarantees that all five
models are trained and evaluated on an identical train/test partition.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- Paths (resolved relative to this file, so notebooks can run from anywhere) ---
MODEL_DIR = Path(__file__).resolve().parent
REPO_ROOT = MODEL_DIR.parent
RAW_PATH = REPO_ROOT / "raw_adult_income.csv"
TEST_CSV_PATH = REPO_ROOT / "test_data.csv"
SAVED_MODELS_DIR = MODEL_DIR / "saved_models"
METRICS_DIR = MODEL_DIR / "metrics"

RANDOM_SEED = 42
POSITIVE_LABEL = ">50K"

NUMERIC_ATTRS = ["age", "fnlwgt", "education_num", "capital_gain",
                 "capital_loss", "hours_per_week"]
CATEGORICAL_ATTRS = ["workclass", "education", "marital_status", "occupation",
                     "relationship", "race", "sex", "native_country"]


def load_raw() -> pd.DataFrame:
    """Return the full raw dataset."""
    return pd.read_csv(RAW_PATH)


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns and impute + one-hot encode categorical columns."""
    numeric_branch = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_branch = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_branch, NUMERIC_ATTRS),
        ("cat", categorical_branch, CATEGORICAL_ATTRS),
    ])


def get_splits(test_size: float = 0.2, export_test_csv: bool = True):
    """Return a stratified (X_train, X_test, y_train, y_test) split.

    When ``export_test_csv`` is True the held-out test rows are written to
    ``test_data.csv`` at the repo root so the Streamlit app can reuse them.
    """
    df = load_raw()
    y = (df["income"].astype(str).str.strip() == POSITIVE_LABEL).astype(int)
    X = df.drop(columns=["income"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED)

    if export_test_csv:
        test_export = X_test.copy()
        test_export["income"] = np.where(y_test.values == 1, ">50K", "<=50K")
        test_export.to_csv(TEST_CSV_PATH, index=False)

    return X_train, X_test, y_train, y_test
