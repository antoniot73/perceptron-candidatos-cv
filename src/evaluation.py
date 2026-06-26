"""Funciones de evaluación del perceptrón."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula exactitud de clasificación.

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.

    Returns:
        Exactitud entre 0 y 1.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener la misma longitud.")
    if len(y_true) == 0:
        raise ValueError("Los vectores no pueden estar vacíos.")
    return float(np.mean(y_true == y_pred))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Calcula conteos TP, TN, FP y FN.

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.

    Returns:
        Diccionario con conteos.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true e y_pred deben tener la misma longitud.")
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def build_results_table(
    df: pd.DataFrame,
    z_values: np.ndarray,
    predictions: np.ndarray,
    target_column: str,
    prediction_column: str,
    z_column: str,
) -> pd.DataFrame:
    """Construye tabla de resultados.

    Args:
        df: DataFrame base.
        z_values: Valores z.
        predictions: Predicciones.
        target_column: Columna objetivo.
        prediction_column: Nombre de la predicción.
        z_column: Nombre de columna z.

    Returns:
        DataFrame ampliado.
    """
    if len(df) != len(z_values) or len(df) != len(predictions):
        raise ValueError("df, z_values y predictions deben tener la misma longitud.")
    results = df.copy()
    results[z_column] = z_values
    results[prediction_column] = predictions
    results[f"acierto_{prediction_column}"] = (
        results[target_column].astype(int) == results[prediction_column].astype(int)
    )
    return results


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    """Guarda un DataFrame como CSV.

    Args:
        df: DataFrame a guardar.
        path: Ruta destino.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    LOGGER.info("Tabla guardada en: %s", output_path)
