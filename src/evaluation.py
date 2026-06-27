"""Funciones de evaluación y persistencia de resultados.

Este módulo contiene utilidades para calcular exactitud, construir tablas de
resultados, obtener conteos de clasificación y guardar DataFrames en archivos CSV
sin mostrar rutas absolutas locales en consola o notebook.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula la exactitud de clasificación del perceptrón.

    En el pipeline de evaluación, esta métrica resume la proporción de registros
    clasificados correctamente. Se utiliza para comparar el desempeño del modelo
    sin entrenamiento contra el modelo entrenado y para evaluar candidatos
    nuevos.

    Args:
        y_true: Vector con etiquetas reales o esperadas.
        y_pred: Vector con etiquetas predichas por el perceptrón.

    Returns:
        Proporción de aciertos en el intervalo [0.0, 1.0].

    Raises:
        ValueError: Si los vectores tienen longitudes diferentes o están vacíos.
    """
    if len(y_true) == 0:
        raise ValueError("El vector y_true no puede estar vacío.")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener la misma longitud.")

    return float(np.mean(y_true == y_pred))


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Obtiene conteos básicos de clasificación binaria.

    Esta función complementa la exactitud al distinguir aciertos y errores según
    la clase. En la práctica permite interpretar si el perceptrón genera falsos
    positivos, falsos negativos o clasificaciones correctas para las clases
    ``pasa`` y ``no pasa``.

    Args:
        y_true: Vector con etiquetas reales o esperadas.
        y_pred: Vector con etiquetas predichas por el modelo.

    Returns:
        Diccionario con verdaderos positivos, verdaderos negativos, falsos
        positivos y falsos negativos.

    Raises:
        ValueError: Si los vectores tienen longitudes diferentes.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener la misma longitud.")

    true_positive = int(np.sum((y_true == 1) & (y_pred == 1)))
    true_negative = int(np.sum((y_true == 0) & (y_pred == 0)))
    false_positive = int(np.sum((y_true == 0) & (y_pred == 1)))
    false_negative = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {
        "VP": true_positive,
        "VN": true_negative,
        "FP": false_positive,
        "FN": false_negative,
    }


def build_results_table(
    df: pd.DataFrame,
    z_values: np.ndarray,
    predictions: np.ndarray,
    target_column: str,
    prediction_column: str,
    z_column: str,
) -> pd.DataFrame:
    """Construye una tabla trazable de resultados del perceptrón.

    La tabla resultante conserva los datos originales e incorpora la suma
    ponderada ``z``, la predicción binaria y una columna de acierto. Esta salida
    es usada en el notebook, en los archivos CSV y en el análisis comparativo
    entre perceptrón sin entrenamiento y perceptrón entrenado.

    Args:
        df: DataFrame base con candidatos y variables del problema.
        z_values: Valores de suma ponderada calculados por el perceptrón.
        predictions: Predicciones binarias generadas por el modelo.
        target_column: Nombre de la columna con la etiqueta real o esperada.
        prediction_column: Nombre de la columna para guardar predicciones.
        z_column: Nombre de la columna para guardar valores ``z``.

    Returns:
        DataFrame con columnas originales más ``z``, predicción y acierto.

    Raises:
        ValueError: Si falta la columna objetivo o las longitudes no coinciden.
    """
    if target_column not in df.columns:
        raise ValueError(f"No existe la columna objetivo: {target_column}")

    if len(df) != len(z_values) or len(df) != len(predictions):
        raise ValueError("df, z_values y predictions deben tener la misma longitud.")

    results = df.copy()
    results[z_column] = z_values
    results[prediction_column] = predictions
    results["acierto"] = (results[target_column].to_numpy(dtype=int) == predictions).astype(int)

    return results


def _display_relative_path(path: Path) -> str:
    """Convierte una ruta de salida a una ruta relativa legible.

    Args:
        path: Ruta absoluta o relativa del archivo.

    Returns:
        Ruta relativa en formato portable para consola/notebook.
    """
    path = Path(path)
    parts = path.parts

    if "outputs" in parts:
        index = parts.index("outputs")
        return str(Path(*parts[index:])).replace("\\", "/")

    return path.name


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    """Guarda una tabla de resultados en CSV con salida de ruta relativa.

    Esta función cierra la etapa de reporte tabular del pipeline. Crea la carpeta
    de salida cuando sea necesario, persiste los resultados y registra una ruta
    relativa para evitar exponer rutas absolutas del equipo local en consola,
    notebook o HTML.

    Args:
        df: DataFrame a guardar.
        output_path: Ruta de salida del archivo CSV.

    Raises:
        ValueError: Si el DataFrame está vacío.
        OSError: Si ocurre un error al crear carpetas o guardar el archivo.
    """
    if df.empty:
        raise ValueError("No se puede guardar un DataFrame vacío.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    logger.info("Tabla guardada en: %s", _display_relative_path(output_path))
