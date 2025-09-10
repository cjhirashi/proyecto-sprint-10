"""
preprocessing.py
Utilidades de preprocesamiento para el proyecto Sprint 10 (Beta Bank).

Incluye:
- Carga del dataset
- Tratamiento de nulos en Tenure (NaN -> 0)
- Selección de variables (features/target)
- One-Hot Encoding (Geography, Gender)
- Escalado con StandardScaler
- Partición estratificada en train/valid/test
- Balanceo manual (oversampling / undersampling)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE: int = 42


@dataclass
class SplitData:
    """Contenedor tipado para devolver los splits."""
    features_train: pd.DataFrame
    features_valid: pd.DataFrame
    features_test: pd.DataFrame
    target_train: pd.Series
    target_valid: pd.Series
    target_test: pd.Series


def load_data(csv_path: str) -> pd.DataFrame:
    """Carga el archivo CSV en un DataFrame."""
    df = pd.read_csv(csv_path)
    return df


def quality_report(df: pd.DataFrame) -> Dict[str, object]:
    """Devuelve un pequeño diagnóstico de calidad."""
    return {
        "shape": df.shape,
        "nulls": df.isnull().sum().to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def treat_tenure_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reemplaza NaN en Tenure por 0 (clientes con <1 año), respetando la semántica del campo.
    Devuelve un nuevo DataFrame (no modifica el original).
    """
    df_out = df.copy()
    if "Tenure" in df_out.columns:
        df_out["Tenure"] = df_out["Tenure"].fillna(0)
    return df_out


def select_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Elimina identificadores que no aportan al modelo y define features/target.
    target = 'Exited'
    """
    cols_to_drop = [c for c in ["RowNumber", "CustomerId", "Surname"] if c in df.columns]
    df_model = df.drop(columns=cols_to_drop)

    target = df_model["Exited"]
    features = df_model.drop(columns=["Exited"])
    return features, target


def encode_categoricals(features: pd.DataFrame) -> pd.DataFrame:
    """
    One-Hot Encoding de Geography y Gender con drop_first=True
    para evitar multicolinealidad.
    """
    cols = [c for c in ["Geography", "Gender"] if c in features.columns]
    features_encoded = pd.get_dummies(features, columns=cols, drop_first=True)
    return features_encoded


def scale_features(features_encoded: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Escalado estándar (media 0, desv. 1) sobre todas las columnas numéricas
    (en este punto ya son todas numéricas).
    Devuelve (DataFrame escalado, scaler ajustado).
    """
    scaler = StandardScaler()
    arr = scaler.fit_transform(features_encoded.values)
    features_scaled = pd.DataFrame(arr, columns=features_encoded.columns, index=features_encoded.index)
    return features_scaled, scaler


def split_train_valid_test(
    features_scaled: pd.DataFrame,
    target: pd.Series,
    valid_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> SplitData:
    """
    Partición estratificada 60/20/20 por defecto (train/valid/test).
    Primero separa test, luego valid desde el bloque temporal.
    """
    # Train vs temp (train 60%, temp 40%)
    features_train, features_temp, target_train, target_temp = train_test_split(
        features_scaled, target, test_size=(valid_size + test_size),
        stratify=target, random_state=random_state
    )
    # Valid vs test (20% / 20% del total)
    valid_fraction = valid_size / (valid_size + test_size)
    features_valid, features_test, target_valid, target_test = train_test_split(
        features_temp, target_temp, test_size=(1 - valid_fraction),
        stratify=target_temp, random_state=random_state
    )
    return SplitData(
        features_train=features_train, features_valid=features_valid, features_test=features_test,
        target_train=target_train, target_valid=target_valid, target_test=target_test
    )


# ==========================
# Balanceo manual de clases
# ==========================

def oversample_minority(features_train: pd.DataFrame, target_train: pd.Series, factor: int = 2
                        ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Sobremuestreo simple de la clase minoritaria duplicando/replicando filas.
    - factor=2 significa duplicar la minoritaria (aprox).
    Nota: no intenta igualar exactamente, pero la acerca de forma sencilla y rápida.
    """
    train_concat = pd.concat([features_train, target_train.rename("Exited")], axis=1)

    minority = train_concat[train_concat["Exited"] == 1]
    majority = train_concat[train_concat["Exited"] == 0]

    # Replicar minoritaria
    minority_oversampled = pd.concat([minority] * factor, axis=0)

    # Unir y barajar
    train_over = pd.concat([majority, minority_oversampled]).sample(frac=1, random_state=RANDOM_STATE)

    features_train_over = train_over.drop(columns=["Exited"])
    target_train_over = train_over["Exited"]
    return features_train_over, target_train_over


def oversample_to_balance(features_train: pd.DataFrame, target_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Variante que intenta acercar la minoritaria al tamaño de la mayoritaria.
    """
    train_concat = pd.concat([features_train, target_train.rename("Exited")], axis=1)

    minority = train_concat[train_concat["Exited"] == 1]
    majority = train_concat[train_concat["Exited"] == 0]

    if len(minority) == 0:
        return features_train.copy(), target_train.copy()

    # Número de réplicas necesarias (redondeo hacia arriba)
    reps = int(np.ceil(len(majority) / max(len(minority), 1)))
    minority_oversampled = pd.concat([minority] * reps, axis=0).head(len(majority))

    train_over = pd.concat([majority, minority_oversampled]).sample(frac=1, random_state=RANDOM_STATE)
    return train_over.drop(columns=["Exited"]), train_over["Exited"]


def undersample_majority(features_train: pd.DataFrame, target_train: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Submuestreo aleatorio de la clase mayoritaria hasta igualar el tamaño de la minoritaria.
    """
    train_concat = pd.concat([features_train, target_train.rename("Exited")], axis=1)

    minority = train_concat[train_concat["Exited"] == 1]
    majority = train_concat[train_concat["Exited"] == 0]

    majority_under = majority.sample(len(minority), random_state=RANDOM_STATE)

    train_under = pd.concat([majority_under, minority]).sample(frac=1, random_state=RANDOM_STATE)
    return train_under.drop(columns=["Exited"]), train_under["Exited"]
