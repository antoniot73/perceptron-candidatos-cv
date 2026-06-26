"""Pruebas unitarias mínimas del perceptrón."""

from __future__ import annotations

import numpy as np

from src.dataset import (
    compute_compliance_indicator,
    compute_expected_label,
    normalize_experience,
    normalize_score,
)
from src.perceptron import PerceptronConfig, RosenblattPerceptron


def test_normalization_functions() -> None:
    """Valida normalización de experiencia y puntaje."""
    assert normalize_experience(2.0) == 0.5
    assert normalize_experience(2.8) == 0.7
    assert normalize_experience(5.0) == 1.0
    assert normalize_score(70) == 0.7
    assert normalize_score(75) == 0.75


def test_expected_label_rule() -> None:
    """Valida regla de mínimo simultáneo por variable."""
    assert compute_compliance_indicator(0.70) == 1
    assert compute_compliance_indicator(0.69) == 0
    assert compute_expected_label(0.70, 0.70) == 1
    assert compute_expected_label(0.80, 0.80) == 1
    assert compute_expected_label(0.60, 0.80) == 0
    assert compute_expected_label(0.80, 0.60) == 0


def test_perceptron_learns_and_rule() -> None:
    """Comprueba que el perceptrón aprende la regla AND de cumplimiento."""
    x_matrix = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ],
        dtype=float,
    )
    y_vector = np.array([0, 0, 0, 1], dtype=int)

    model = RosenblattPerceptron(PerceptronConfig(learning_rate=0.1, max_epochs=100))
    model.train(x_matrix, y_vector)
    predictions = model.predict(x_matrix)

    assert np.array_equal(predictions, y_vector)
