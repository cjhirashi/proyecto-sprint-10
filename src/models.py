"""
models.py
Funciones de entrenamiento y evaluación de modelos (árbol, random forest, regresión logística)
alineadas al flujo del notebook: uso de features/target, métricas F1 y AUC-ROC.
"""

from __future__ import annotations
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def evaluate_model(
    model,
    features_train: pd.DataFrame,
    target_train: pd.Series,
    features_valid: pd.DataFrame,
    target_valid: pd.Series,
) -> Dict[str, float]:
    """
    Entrena el modelo y evalúa en validación.
    Devuelve un diccionario con F1 y AUC-ROC.
    """
    model.fit(features_train, target_train)

    preds_valid = model.predict(features_valid)
    if hasattr(model, "predict_proba"):
        probs_valid = model.predict_proba(features_valid)[:, 1]
    else:
        # fallback para modelos sin predict_proba
        # se usa la predicción como prob surrogate (no ideal, pero evita romper)
        probs_valid = preds_valid.astype(float)

    f1 = f1_score(target_valid, preds_valid)
    auc = roc_auc_score(target_valid, probs_valid)

    return {"f1": float(f1), "auc": float(auc)}


# =======================
# Modelos específicos
# =======================

def decision_tree_baseline(
    max_depth: int = 5,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> DecisionTreeClassifier:
    """Devuelve un DecisionTreeClassifier con hiperparámetros básicos."""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )


def random_forest_baseline(
    n_estimators: int = 100,
    max_depth: int = 10,
    min_samples_split: int = 20,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Devuelve un RandomForestClassifier con configuración estable."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )


def logistic_regression_baseline(
    solver: str = "liblinear",
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Devuelve un modelo de Regresión Logística básico."""
    return LogisticRegression(
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )


# =======================
# Helpers para test final
# =======================

def evaluate_on_test(
    model,
    features_test: pd.DataFrame,
    target_test: pd.Series,
) -> Dict[str, float]:
    """
    Evalúa un modelo entrenado sobre el conjunto de prueba (test).
    Devuelve F1 y AUC-ROC.
    """
    preds = model.predict(features_test)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_test)[:, 1]
    else:
        probs = preds.astype(float)

    f1 = f1_score(target_test, preds)
    auc = roc_auc_score(target_test, probs)
    return {"f1": float(f1), "auc": float(auc)}
