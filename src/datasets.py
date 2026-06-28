
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def load_heart():
    df = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"), encoding="utf-8-sig")
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y, list(X.columns)


def load_diabetes_uci():
    df = pd.read_csv(os.path.join(DATA_DIR, "diabetes.csv"))
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return X, y, list(X.columns)


def load_breast_cancer_uci():
    data = load_breast_cancer(as_frame=True)
    X = data.data
    # sklearn encodes 0=malignant, 1=benign -- flip so 1 = "has disease" (malignant),
    # consistent with the other two datasets where 1 = disease present
    y = 1 - data.target
    return X, y, list(X.columns)


DATASETS = {
    "heart_disease": load_heart,
    "diabetes": load_diabetes_uci,
    "breast_cancer": load_breast_cancer_uci,
}
