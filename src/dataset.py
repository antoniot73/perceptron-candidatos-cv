"""Carga, validación y preparación de datasets del proyecto.

Este módulo concentra la capa de entrada y validación de datos. Su función
dentro del pipeline es garantizar que los archivos CSV usados por el perceptrón
contengan las columnas requeridas, rangos numéricos válidos y etiquetas
coherentes con la regla académica de clasificación:

    x1 >= 0.70 y x2 >= 0.70 -> y = 1
    en otro caso            -> y = 0

También contiene funciones puras para normalizar variables y calcular etiquetas
esperadas, lo que facilita la reproducibilidad, las pruebas unitarias y la
trazabilidad del experimento.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd

LOGGER = logging.getLogger(__name__)

MIN_APPROVAL_THRESHOLD: float = 0.70

REQUIRED_TRAIN_COLUMNS: tuple[str, ...] = (
    "id", "candidato", "anios_experiencia", "puntaje_tecnico",
    "x1_experiencia_norm", "x2_puntaje_norm",
    "cumple_experiencia", "cumple_puntaje",
    "suma_cumplimiento", "y_pasa_entrevista",
)

REQUIRED_TEST_COLUMNS: tuple[str, ...] = (
    "id", "candidato", "anios_experiencia", "puntaje_tecnico",
    "x1_experiencia_norm", "x2_puntaje_norm",
    "cumple_experiencia", "cumple_puntaje",
    "suma_cumplimiento", "y_esperada",
)


def configure_logging(level: int = logging.INFO) -> None:
    """Configura la bitácora básica de ejecución del proyecto.

    Esta función prepara el registro de eventos para que el pipeline informe
    acciones relevantes, como la carga de datasets y la validación de registros.
    Se usa tanto en ejecución local como desde el notebook para mantener una
    salida trazable y homogénea.

    Args:
        level: Nivel mínimo de logging que será mostrado en consola.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def format_dataset_path(path: str | Path) -> str:
    """Devuelve una ruta relativa legible para mensajes de consola.

    Args:
        path: Ruta absoluta o relativa del dataset.

    Returns:
        Ruta breve en formato POSIX, por ejemplo ``data/candidatos_30.csv``.
    """
    dataset_path = Path(path)
    return f"data/{dataset_path.name}"


def normalize_experience(years: float, max_years: float = 4.0) -> float:
    """Normaliza años de experiencia laboral al intervalo [0, 1].

    Dentro del pipeline, esta función transforma la medición original de
    experiencia, expresada en años, en una variable numérica comparable para el
    perceptrón. El valor de referencia ``max_years`` representa la experiencia
    máxima considerada para la práctica; valores superiores se saturan en 1.0.

    Args:
        years: Años de experiencia laboral. Debe ser mayor o igual que cero.
        max_years: Valor de referencia que equivale a 1.0 en escala normalizada.

    Returns:
        Experiencia normalizada en el intervalo cerrado [0, 1].

    Raises:
        ValueError: Si ``years`` es negativo o ``max_years`` no es positivo.
    """
    if max_years <= 0:
        raise ValueError("max_years debe ser mayor que cero.")
    if years < 0:
        raise ValueError("Los años de experiencia no pueden ser negativos.")
    return min(years / max_years, 1.0)


def normalize_score(score: float) -> float:
    """Normaliza el puntaje técnico al intervalo [0, 1].

    Esta transformación convierte la evaluación técnica original, expresada en
    una escala de 0 a 100 puntos, en una variable compatible con el modelo
    lineal del perceptrón y con la regla de cumplimiento mínimo de la práctica.

    Args:
        score: Puntaje técnico original. Debe estar entre 0 y 100.

    Returns:
        Puntaje técnico normalizado en el intervalo cerrado [0, 1].

    Raises:
        ValueError: Si el puntaje está fuera del rango permitido.
    """
    if score < 0 or score > 100:
        raise ValueError("El puntaje técnico debe estar entre 0 y 100.")
    return score / 100.0


def compute_compliance_indicator(value: float, threshold: float = MIN_APPROVAL_THRESHOLD) -> int:
    """Calcula si una variable normalizada cumple el mínimo aprobatorio.

    Args:
        value: Valor normalizado entre 0 y 1.
        threshold: Mínimo aprobatorio individual.

    Returns:
        1 si value >= threshold; si no, 0.
    """
    if not 0 <= value <= 1:
        raise ValueError("El valor normalizado debe estar entre 0 y 1.")
    if not 0 <= threshold <= 1:
        raise ValueError("El umbral debe estar entre 0 y 1.")
    return int(value >= threshold)


def compute_expected_label(
    x1: float,
    x2: float,
    threshold: float = MIN_APPROVAL_THRESHOLD,
) -> int:
    """Calcula la etiqueta esperada mediante la regla de mínimo simultáneo.

    Esta función define la lógica supervisada que el perceptrón debe aprender.
    La salida positiva solo ocurre cuando las dos variables normalizadas cumplen
    el umbral mínimo. En términos lógicos, equivale a una compuerta AND sobre
    los indicadores de cumplimiento de experiencia y puntaje técnico.

    Args:
        x1: Experiencia laboral normalizada.
        x2: Puntaje técnico normalizado.
        threshold: Mínimo aprobatorio individual para cada variable.

    Returns:
        1 si ``x1 >= threshold`` y ``x2 >= threshold``; en caso contrario, 0.
    """
    cumple_x1 = compute_compliance_indicator(x1, threshold)
    cumple_x2 = compute_compliance_indicator(x2, threshold)
    return int(cumple_x1 == 1 and cumple_x2 == 1)


def add_compliance_columns(
    df: pd.DataFrame,
    threshold: float = MIN_APPROVAL_THRESHOLD,
) -> pd.DataFrame:
    """Agrega columnas binarias de cumplimiento mínimo.

    Args:
        df: DataFrame con variables normalizadas.
        threshold: Mínimo aprobatorio individual.

    Returns:
        Copia del DataFrame con columnas de cumplimiento.
    """
    validate_columns(df, ("x1_experiencia_norm", "x2_puntaje_norm"))
    output = df.copy()
    output["cumple_experiencia"] = output["x1_experiencia_norm"].apply(
        lambda value: compute_compliance_indicator(float(value), threshold)
    )
    output["cumple_puntaje"] = output["x2_puntaje_norm"].apply(
        lambda value: compute_compliance_indicator(float(value), threshold)
    )
    return output


def load_csv(path: str | Path) -> pd.DataFrame:
    """Carga un archivo CSV validando ruta y extensión.

    Args:
        path: Ruta del archivo CSV.

    Returns:
        DataFrame cargado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si el archivo no tiene extensión CSV.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {format_dataset_path(csv_path)}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"El archivo debe ser CSV: {format_dataset_path(csv_path)}")

    LOGGER.info("Cargando dataset: %s", format_dataset_path(csv_path))
    return pd.read_csv(csv_path)


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    """Valida columnas obligatorias.

    Args:
        df: DataFrame a validar.
        required_columns: Columnas requeridas.

    Raises:
        ValueError: Si faltan columnas.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas obligatorias: {missing}")


def validate_numeric_ranges(df: pd.DataFrame) -> None:
    """Valida rangos numéricos principales.

    Args:
        df: DataFrame con columnas numéricas.

    Raises:
        ValueError: Si encuentra valores fuera de rango.
    """
    if (df["anios_experiencia"] < 0).any():
        raise ValueError("Existen años de experiencia negativos.")
    if ((df["puntaje_tecnico"] < 0) | (df["puntaje_tecnico"] > 100)).any():
        raise ValueError("Existen puntajes técnicos fuera de [0, 100].")
    for col in ("x1_experiencia_norm", "x2_puntaje_norm"):
        if ((df[col] < 0) | (df[col] > 1)).any():
            raise ValueError(f"La columna {col} tiene valores fuera de [0, 1].")
    for col in ("cumple_experiencia", "cumple_puntaje"):
        if col in df.columns and not set(df[col].unique()).issubset({0, 1}):
            raise ValueError(f"La columna {col} solo puede contener 0 y 1.")


def validate_expected_labels(
    df: pd.DataFrame,
    target_column: str,
    threshold: float = MIN_APPROVAL_THRESHOLD,
) -> None:
    """Valida que la etiqueta esperada coincida con el mínimo simultáneo.

    Args:
        df: DataFrame validado.
        target_column: Nombre de la columna objetivo.
        threshold: Mínimo aprobatorio individual.

    Raises:
        ValueError: Si alguna etiqueta no coincide con la regla esperada.
    """
    expected = df.apply(
        lambda row: compute_expected_label(
            float(row["x1_experiencia_norm"]),
            float(row["x2_puntaje_norm"]),
            threshold,
        ),
        axis=1,
    )
    mismatches = df.index[df[target_column].astype(int) != expected.astype(int)].tolist()
    if mismatches:
        raise ValueError(
            f"La columna {target_column} contiene etiquetas inconsistentes "
            f"con la regla de mínimo simultáneo en filas: {mismatches}"
        )


def load_training_dataset(path: str | Path) -> pd.DataFrame:
    """Carga y valida el dataset de entrenamiento del perceptrón.

    Esta función representa la entrada principal de datos supervisados del
    pipeline. Verifica que el archivo exista, que contenga las columnas
    requeridas, que los valores numéricos estén dentro de rangos válidos y que
    la etiqueta ``y_pasa_entrevista`` sea coherente con la regla de clasificación.

    Args:
        path: Ruta del archivo CSV con los candidatos de entrenamiento.

    Returns:
        DataFrame validado, listo para extraer variables de entrada y etiqueta.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas, hay rangos inválidos o etiquetas
            inconsistentes.
    """
    df = load_csv(path)
    validate_columns(df, REQUIRED_TRAIN_COLUMNS)
    validate_numeric_ranges(df)
    validate_expected_labels(df, "y_pasa_entrevista")
    LOGGER.info("Dataset de entrenamiento validado: %d registros.", len(df))
    return df


def load_test_dataset(path: str | Path) -> pd.DataFrame:
    """Carga y valida el dataset de candidatos nuevos.

    Esta función prepara los casos externos al entrenamiento que se utilizan para
    comprobar si el perceptrón entrenado aplica correctamente la regla aprendida
    a registros no vistos durante el ajuste de pesos y bias.

    Args:
        path: Ruta del archivo CSV con candidatos de prueba.

    Returns:
        DataFrame validado, listo para evaluación del modelo entrenado.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si faltan columnas, hay rangos inválidos o etiquetas
            inconsistentes.
    """
    df = load_csv(path)
    validate_columns(df, REQUIRED_TEST_COLUMNS)
    validate_numeric_ranges(df)
    validate_expected_labels(df, "y_esperada")
    LOGGER.info("Dataset de prueba validado: %d registros.", len(df))
    return df
