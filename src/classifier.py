import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def train(
    X_train: pd.DataFrame, y_train: pd.Series
) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_iter=300,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(
    model: HistGradientBoostingClassifier, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    return {
        "auc": roc_auc_score(y_test, y_prob),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "cm": confusion_matrix(y_test, y_pred),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }


def compute_shap(
    model: HistGradientBoostingClassifier, X: pd.DataFrame, n_samples: int = 300
) -> tuple:
    X_sample = X.sample(min(n_samples, len(X)), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer(X_sample)
    return explainer, shap_vals, X_sample
