"""Ejecución local completa del proyecto.

Ejecutar:
    python main.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset import MIN_APPROVAL_THRESHOLD, configure_logging, load_test_dataset, load_training_dataset
from src.evaluation import accuracy_score, build_results_table, confusion_counts, save_dataframe
from src.perceptron import PerceptronConfig, RosenblattPerceptron
from src.visualization import (
    plot_accuracy_comparison,
    plot_error_history,
    plot_learned_and_boundary,
    plot_minimum_criteria,
)

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_PATH = PROJECT_ROOT / "data" / "candidatos_30.csv"
TEST_PATH = PROJECT_ROOT / "data" / "candidatos_prueba.csv"
OUTPUT_TABLES = PROJECT_ROOT / "outputs" / "tablas"
OUTPUT_GRAPHS = PROJECT_ROOT / "outputs" / "graficas"

FEATURE_COLUMNS: list[str] = ["cumple_experiencia", "cumple_puntaje"]


def get_features_and_target(df: pd.DataFrame, target_column: str) -> tuple[np.ndarray, np.ndarray]:
    """Extrae matriz X y vector y desde un DataFrame.

    Args:
        df: DataFrame fuente.
        target_column: Columna objetivo.

    Returns:
        Tupla (X, y).
    """
    missing_features = [column for column in FEATURE_COLUMNS if column not in df.columns]
    if missing_features:
        raise ValueError(f"Faltan columnas de entrada del perceptrón: {missing_features}")
    if target_column not in df.columns:
        raise ValueError(f"No existe la columna objetivo: {target_column}")

    x_matrix = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y_vector = df[target_column].to_numpy(dtype=int)
    return x_matrix, y_vector


def run_project() -> None:
    """Ejecuta el flujo completo del proyecto local.

    Carga datos, evalúa el perceptrón sin entrenamiento, entrena el modelo,
    prueba candidatos nuevos, guarda tablas y genera gráficas.
    """
    configure_logging()
    print("\n=== Proyecto: Perceptrón de Rosenblatt para selección de candidatos ===\n")

    train_df = load_training_dataset(TRAIN_PATH)
    test_df = load_test_dataset(TEST_PATH)
    x_train, y_train = get_features_and_target(train_df, "y_pasa_entrevista")

    print(f"Candidatos de entrenamiento cargados: {len(train_df)}")
    print(f"Candidatos de prueba cargados: {len(test_df)}")
    print("\nEntradas reales del perceptrón:")
    print("c1 = cumple experiencia mínima, calculada desde x1 >= 0.70")
    print("c2 = cumple puntaje técnico mínimo, calculada desde x2 >= 0.70")

    print("\n--- Prueba sin entrenamiento ---")
    untrained_model = RosenblattPerceptron.with_zero_weights(
        n_features=2,
        learning_rate=0.1,
        max_epochs=100,
        initial_bias=0.0,
    )
    initial_weights = untrained_model.weights.copy()
    initial_bias = untrained_model.bias
    print(f"Pesos iniciales: {initial_weights}")
    print(f"Bias inicial: {initial_bias:.4f}")

    z_initial = untrained_model.decision_scores(x_train)
    y_initial = untrained_model.predict(x_train)
    initial_accuracy = accuracy_score(y_train, y_initial)
    print(f"Exactitud inicial: {initial_accuracy:.2%}")
    print(f"Conteos iniciales: {confusion_counts(y_train, y_initial)}")

    initial_results = build_results_table(
        train_df,
        z_initial,
        y_initial,
        target_column="y_pasa_entrevista",
        prediction_column="y_pred_sin_entrenamiento",
        z_column="z_sin_entrenamiento",
    )
    save_dataframe(initial_results, OUTPUT_TABLES / "resultados_sin_entrenamiento.csv")

    print("\n--- Entrenamiento ---")
    model = RosenblattPerceptron(
        PerceptronConfig(learning_rate=0.1, max_epochs=100, initial_bias=0.0)
    )
    model.initialize_weights(n_features=2, mode="zeros")
    errors = model.train(x_train, y_train)
    weight_change = model.weights - initial_weights
    bias_change = model.bias - initial_bias

    print(f"Errores por época: {errors}")
    print(f"Pesos finales: {model.weights}")
    print(f"Bias final: {model.bias:.4f}")
    print(f"Cambio de pesos: {weight_change}")
    print(f"Cambio del bias: {bias_change:.4f}")

    print("\n--- Prueba con entrenamiento ---")
    z_final = model.decision_scores(x_train)
    y_final = model.predict(x_train)
    final_accuracy = accuracy_score(y_train, y_final)
    print(f"Exactitud final: {final_accuracy:.2%}")
    print(f"Conteos finales: {confusion_counts(y_train, y_final)}")

    final_results = build_results_table(
        train_df,
        z_final,
        y_final,
        target_column="y_pasa_entrevista",
        prediction_column="y_pred_entrenado",
        z_column="z_entrenado",
    )
    final_results["y_pred_sin_entrenamiento"] = y_initial
    final_results["z_sin_entrenamiento"] = z_initial
    save_dataframe(final_results, OUTPUT_TABLES / "resultados_con_entrenamiento.csv")

    print("\n--- Prueba con candidatos nuevos ---")
    x_test, y_test = get_features_and_target(test_df, "y_esperada")
    z_test = model.decision_scores(x_test)
    y_test_pred = model.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"Exactitud en candidatos nuevos: {test_accuracy:.2%}")

    test_results = build_results_table(
        test_df,
        z_test,
        y_test_pred,
        target_column="y_esperada",
        prediction_column="y_pred_modelo",
        z_column="z_modelo",
    )
    save_dataframe(test_results, OUTPUT_TABLES / "resultados_candidatos_prueba.csv")

    print("\n--- Visualizaciones ---")
    plot_error_history(errors, OUTPUT_GRAPHS / "errores_por_epoca.png")
    plot_minimum_criteria(
        train_df,
        OUTPUT_GRAPHS / "criterios_minimos_aprobacion.png",
        threshold=MIN_APPROVAL_THRESHOLD,
    )
    plot_learned_and_boundary(
        model.weights,
        model.bias,
        OUTPUT_GRAPHS / "frontera_and_aprendida.png",
    )
    plot_accuracy_comparison(initial_accuracy, final_accuracy, OUTPUT_GRAPHS / "comparacion_exactitud.png")

    print("\nProyecto ejecutado correctamente.")
    print(f"Tablas guardadas en: {OUTPUT_TABLES}")
    print(f"Gráficas guardadas en: {OUTPUT_GRAPHS}")


if __name__ == "__main__":
    run_project()
