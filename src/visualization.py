"""Visualizaciones del pipeline del perceptrón.

Este módulo contiene la capa de comunicación visual del proyecto. Sus funciones
generan gráficas para interpretar el aprendizaje del perceptrón: evolución de
errores, comparación de exactitud, criterios mínimos de aprobación y frontera de
decisión aprendida.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_error_history(errors: list[int], output_path: str | Path | None = None) -> None:
    """Grafica la evolución de errores durante el entrenamiento.

    Esta visualización permite comprobar si el perceptrón reduce sus errores a
    medida que actualiza pesos y bias. En un problema linealmente separable,
    como el de esta práctica, la curva debe tender a cero cuando el modelo
    converge.

    Args:
        errors: Lista de errores cometidos en cada época.
        output_path: Ruta opcional para guardar la figura.

    Raises:
        ValueError: Si la lista de errores está vacía.
    """
    if not errors:
        raise ValueError("La lista de errores no puede estar vacía.")
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(errors) + 1)
    ax.plot(epochs, errors, marker="o")
    ax.set_title("Errores por época durante el entrenamiento")
    ax.set_xlabel("Época")
    ax.set_ylabel("Número de errores")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_minimum_criteria(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
    threshold: float = 0.70,
    title: str = "Criterios mínimos de aprobación",
) -> None:
    """Grafica los candidatos frente a los criterios mínimos de aprobación.

    La figura muestra las variables normalizadas ``x1`` y ``x2`` junto con los
    umbrales de cumplimiento. Sirve para explicar visualmente que la clase
    positiva solo ocurre en la zona donde ambas variables alcanzan o superan el
    mínimo aprobatorio.

    Args:
        df: DataFrame con variables normalizadas y etiqueta real.
        output_path: Ruta opcional para guardar la figura.
        threshold: Mínimo aprobatorio individual.
        title: Título de la gráfica.

    Raises:
        ValueError: Si faltan columnas requeridas o el umbral está fuera de [0, 1].
    """
    required = {"x1_experiencia_norm", "x2_puntaje_norm"}
    if not required.issubset(df.columns):
        raise ValueError(f"El DataFrame debe contener columnas: {required}")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1.")

    fig, ax = plt.subplots(figsize=(8, 6))
    target_col = "y_pasa_entrevista" if "y_pasa_entrevista" in df.columns else "y_esperada"

    for label in sorted(df[target_col].unique()):
        subset = df[df[target_col] == label]
        ax.scatter(
            subset["x1_experiencia_norm"],
            subset["x2_puntaje_norm"],
            label=f"Clase real {label}",
            s=70,
            edgecolor="black",
            alpha=0.8,
        )

    ax.axvline(threshold, linestyle="--", linewidth=2, label=f"Umbral x1 = {threshold:.2f}")
    ax.axhline(threshold, linestyle="--", linewidth=2, label=f"Umbral x2 = {threshold:.2f}")

    ax.text(
        threshold + 0.02,
        1.02,
        "Zona aprobatoria:\nx1 ≥ 0.70 y x2 ≥ 0.70",
        ha="left",
        va="top",
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title, pad=14)
    ax.set_xlabel("x1: experiencia laboral normalizada")
    ax.set_ylabel("x2: puntaje técnico normalizado")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_learned_and_boundary(
    weights: np.ndarray,
    bias: float,
    output_path: str | Path | None = None,
    title: str = "Frontera aprendida sobre variables binarias de cumplimiento",
) -> None:
    """Grafica la frontera de decisión aprendida por el perceptrón.

    La gráfica se construye sobre las variables binarias ``c1`` y ``c2`` y
    permite verificar si el modelo aprendió una regla equivalente a la compuerta
    lógica AND. La frontera se obtiene a partir de los pesos y el bias finales.

    Args:
        weights: Pesos del perceptrón entrenado.
        bias: Sesgo del perceptrón entrenado.
        output_path: Ruta opcional para guardar la figura.
        title: Título de la gráfica.

    Raises:
        ValueError: Si los pesos no corresponden a exactamente dos variables.
    """
    if weights.shape[0] != 2:
        raise ValueError("La gráfica solo soporta dos variables.")

    points = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    labels = np.array([0, 0, 0, 1], dtype=int)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label in sorted(set(labels)):
        subset = points[labels == label]
        ax.scatter(
            subset[:, 0],
            subset[:, 1],
            label=f"Clase AND {label}",
            s=100,
            edgecolor="black",
            alpha=0.85,
        )

    x_values = np.linspace(-0.1, 1.1, 100)
    if abs(weights[1]) > 1e-12:
        y_values = -(weights[0] * x_values + bias) / weights[1]
        ax.plot(x_values, y_values, linewidth=2, label="Frontera aprendida")
    elif abs(weights[0]) > 1e-12:
        x_boundary = -bias / weights[0]
        ax.axvline(x_boundary, linewidth=2, label="Frontera aprendida")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_title(title, pad=14)
    ax.set_xlabel("c1: cumple experiencia mínima")
    ax.set_ylabel("c2: cumple puntaje técnico mínimo")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_accuracy_comparison(
    initial_accuracy: float,
    final_accuracy: float,
    output_path: str | Path | None = None,
) -> None:
    """Grafica la exactitud antes y después del entrenamiento.

    Esta visualización resume el impacto del aprendizaje supervisado. Compara el
    desempeño del perceptrón sin pesos ajustados contra el modelo entrenado con
    la regla de Rosenblatt.

    Args:
        initial_accuracy: Exactitud antes del entrenamiento. Debe estar entre 0 y 1.
        final_accuracy: Exactitud después del entrenamiento. Debe estar entre 0 y 1.
        output_path: Ruta opcional para guardar la figura.

    Raises:
        ValueError: Si alguna exactitud está fuera del intervalo [0, 1].
    """
    values = [initial_accuracy, final_accuracy]
    labels = ["Sin entrenamiento", "Con entrenamiento"]

    for value in values:
        if not 0 <= value <= 1:
            raise ValueError("Las exactitudes deben estar en el rango [0, 1].")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bars = ax.bar(labels, values)

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Exactitud")
    ax.set_title("Comparación de exactitud", pad=14)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 0.025, 1.10),
            f"{value:.2%}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()

    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
