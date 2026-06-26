"""Carga, validación y preparación de datasets."""

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
    """Configura logging básico para ejecución local.

    Args:
        level: Nivel mínimo de logging.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def normalize_experience(years: float, max_years: float = 4.0) -> float:
    """Normaliza años de experiencia al intervalo [0, 1].

    Args:
        years: Años de experiencia laboral, de 0 a 4 años o más.
        max_years: Valor de referencia equivalente a 1.0.

    Returns:
        Experiencia normalizada.

    Raises:
        ValueError: Si los valores están fuera de rango.
    """
    if max_years <= 0:
        raise ValueError("max_years debe ser mayor que cero.")
    if years < 0:
        raise ValueError("Los años de experiencia no pueden ser negativos.")
    return min(years / max_years, 1.0)


def normalize_score(score: float) -> float:
    """Normaliza puntaje técnico al intervalo [0, 1].

    Args:
        score: Puntaje entre 0 y 100.

    Returns:
        Puntaje normalizado.

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
    """Calcula etiqueta esperada con regla de mínimo simultáneo.

    Args:
        x1: Experiencia laboral normalizada.
        x2: Puntaje técnico normalizado.
        threshold: Mínimo aprobatorio individual para cada variable.

    Returns:
        1 si x1 >= threshold y x2 >= threshold; si no, 0.
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
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {csv_path}")
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"El archivo debe ser CSV: {csv_path}")
    LOGGER.info("Cargando dataset: %s", csv_path)
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
    """Carga y valida dataset de entrenamiento.

    Args:
        path: Ruta del CSV.

    Returns:
        DataFrame validado.
    """
    df = load_csv(path)
    validate_columns(df, REQUIRED_TRAIN_COLUMNS)
    validate_numeric_ranges(df)
    validate_expected_labels(df, "y_pasa_entrevista")
    LOGGER.info("Dataset de entrenamiento validado: %d registros.", len(df))
    return df


def load_test_dataset(path: str | Path) -> pd.DataFrame:
    """Carga y valida dataset de prueba.

    Args:
        path: Ruta del CSV.

    Returns:
        DataFrame validado.
    """
    df = load_csv(path)
    validate_columns(df, REQUIRED_TEST_COLUMNS)
    validate_numeric_ranges(df)
    validate_expected_labels(df, "y_esperada")
    LOGGER.info("Dataset de prueba validado: %d registros.", len(df))
    return df
