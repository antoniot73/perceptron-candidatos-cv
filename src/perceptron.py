"""Implementación del perceptrón de Frank Rosenblatt."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PerceptronConfig:
    """Configuración del perceptrón.

    Attributes:
        learning_rate: Tasa de aprendizaje.
        max_epochs: Número máximo de épocas.
        initial_bias: Sesgo inicial.
        random_seed: Semilla opcional para inicialización aleatoria.
    """

    learning_rate: float = 0.1
    max_epochs: int = 100
    initial_bias: float = 0.0
    random_seed: int | None = 42

    def __post_init__(self) -> None:
        """Valida parámetros de configuración."""
        if self.learning_rate <= 0:
            raise ValueError("learning_rate debe ser mayor que cero.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs debe ser mayor que cero.")


class RosenblattPerceptron:
    """Perceptrón binario de Rosenblatt para clasificación lineal."""

    def __init__(self, config: PerceptronConfig | None = None) -> None:
        """Inicializa el perceptrón.

        Args:
            config: Configuración del modelo.
        """
        self._config = config or PerceptronConfig()
        self._weights: np.ndarray | None = None
        self._bias: float = self._config.initial_bias
        self._errors_per_epoch: list[int] = []

    @property
    def config(self) -> PerceptronConfig:
        """Devuelve configuración del modelo."""
        return self._config

    @property
    def weights(self) -> np.ndarray:
        """Devuelve copia de pesos actuales."""
        if self._weights is None:
            raise RuntimeError("Los pesos aún no han sido inicializados.")
        return self._weights.copy()

    @property
    def bias(self) -> float:
        """Devuelve sesgo actual."""
        return float(self._bias)

    @property
    def errors_per_epoch(self) -> list[int]:
        """Devuelve bitácora de errores por época."""
        return list(self._errors_per_epoch)

    @classmethod
    def with_zero_weights(
        cls,
        n_features: int,
        learning_rate: float = 0.1,
        max_epochs: int = 100,
        initial_bias: float = 0.0,
    ) -> "RosenblattPerceptron":
        """Crea un perceptrón con pesos iniciales en cero.

        Args:
            n_features: Número de variables.
            learning_rate: Tasa de aprendizaje.
            max_epochs: Máximo de épocas.
            initial_bias: Sesgo inicial.

        Returns:
            Modelo inicializado.
        """
        model = cls(PerceptronConfig(learning_rate, max_epochs, initial_bias, None))
        model.initialize_weights(n_features=n_features, mode="zeros")
        return model

    @staticmethod
    def activation(z_value: float) -> int:
        """Función escalón binaria.

        Args:
            z_value: Suma ponderada.

        Returns:
            1 si z_value >= 0; si no, 0.
        """
        return int(z_value >= 0)

    def __repr__(self) -> str:
        """Representación técnica."""
        weights = None if self._weights is None else self._weights.tolist()
        return f"RosenblattPerceptron(weights={weights}, bias={self._bias:.4f})"

    def initialize_weights(self, n_features: int, mode: str = "zeros") -> None:
        """Inicializa pesos.

        Args:
            n_features: Número de variables.
            mode: 'zeros' o 'random'.

        Raises:
            ValueError: Si los parámetros no son válidos.
        """
        if n_features <= 0:
            raise ValueError("n_features debe ser mayor que cero.")
        if mode == "zeros":
            self._weights = np.zeros(n_features, dtype=float)
        elif mode == "random":
            rng = np.random.default_rng(self._config.random_seed)
            self._weights = rng.normal(loc=0.0, scale=0.01, size=n_features)
        else:
            raise ValueError("mode debe ser 'zeros' o 'random'.")
        self._bias = self._config.initial_bias
        LOGGER.info("Pesos inicializados: %s | bias=%.4f", self._weights, self._bias)

    def net_input(self, x_row: np.ndarray) -> float:
        """Calcula z = w·x + b.

        Args:
            x_row: Vector de entrada.

        Returns:
            Suma ponderada.
        """
        if self._weights is None:
            self.initialize_weights(n_features=x_row.shape[0], mode="zeros")
        return float(np.dot(x_row, self._weights) + self._bias)

    def predict_one(self, x_row: np.ndarray) -> int:
        """Predice una observación.

        Args:
            x_row: Vector de entrada.

        Returns:
            Clase 0/1.
        """
        self._validate_vector(x_row)
        return self.activation(self.net_input(x_row))

    def predict(self, x_matrix: np.ndarray) -> np.ndarray:
        """Predice una matriz de observaciones.

        Args:
            x_matrix: Matriz de entrada.

        Returns:
            Vector de predicciones.
        """
        self._validate_matrix(x_matrix)
        if self._weights is None:
            self.initialize_weights(n_features=x_matrix.shape[1], mode="zeros")
        predictions: list[int] = []
        for row in x_matrix:
            predictions.append(self.predict_one(row))
        return np.array(predictions, dtype=int)

    def decision_scores(self, x_matrix: np.ndarray) -> np.ndarray:
        """Calcula valores z para una matriz.

        Args:
            x_matrix: Matriz de entrada.

        Returns:
            Vector de valores z.
        """
        self._validate_matrix(x_matrix)
        if self._weights is None:
            self.initialize_weights(n_features=x_matrix.shape[1], mode="zeros")
        return np.array([self.net_input(row) for row in x_matrix], dtype=float)

    def train(self, x_matrix: np.ndarray, y_vector: np.ndarray) -> list[int]:
        """Entrena el perceptrón con regla de Rosenblatt.

        Args:
            x_matrix: Variables de entrada.
            y_vector: Etiquetas reales 0/1.

        Returns:
            Lista de errores por época.
        """
        self._validate_training_data(x_matrix, y_vector)
        if self._weights is None:
            self.initialize_weights(n_features=x_matrix.shape[1], mode="zeros")

        self._errors_per_epoch = []

        for epoch in range(1, self._config.max_epochs + 1):
            errors = 0
            for x_row, y_real in zip(x_matrix, y_vector):
                y_pred = self.predict_one(x_row)
                error = int(y_real) - y_pred
                if error != 0:
                    self._weights = self._weights + self._config.learning_rate * error * x_row
                    self._bias = self._bias + self._config.learning_rate * error
                    errors += 1

            self._errors_per_epoch.append(errors)
            LOGGER.info("Época %d | errores=%d", epoch, errors)

            if errors == 0:
                LOGGER.info("Convergencia alcanzada en época %d.", epoch)
                break

        return self.errors_per_epoch

    def _validate_vector(self, x_row: np.ndarray) -> None:
        """Valida vector de entrada."""
        if not isinstance(x_row, np.ndarray):
            raise TypeError("x_row debe ser np.ndarray.")
        if x_row.ndim != 1:
            raise ValueError("x_row debe ser un vector unidimensional.")
        if not np.isfinite(x_row).all():
            raise ValueError("x_row contiene valores no finitos.")

    def _validate_matrix(self, x_matrix: np.ndarray) -> None:
        """Valida matriz de entrada."""
        if not isinstance(x_matrix, np.ndarray):
            raise TypeError("x_matrix debe ser np.ndarray.")
        if x_matrix.ndim != 2:
            raise ValueError("x_matrix debe tener dos dimensiones.")
        if x_matrix.shape[0] == 0:
            raise ValueError("x_matrix no puede estar vacía.")
        if not np.isfinite(x_matrix).all():
            raise ValueError("x_matrix contiene valores no finitos.")

    def _validate_training_data(self, x_matrix: np.ndarray, y_vector: np.ndarray) -> None:
        """Valida datos de entrenamiento."""
        self._validate_matrix(x_matrix)
        if not isinstance(y_vector, np.ndarray):
            raise TypeError("y_vector debe ser np.ndarray.")
        if y_vector.ndim != 1:
            raise ValueError("y_vector debe ser un vector.")
        if len(y_vector) != x_matrix.shape[0]:
            raise ValueError("X e y tienen diferente número de registros.")
        valid_values = set(np.unique(y_vector).tolist())
        if not valid_values.issubset({0, 1}):
            raise ValueError("y_vector solo puede contener 0 y 1.")
