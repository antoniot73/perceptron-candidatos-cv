"""
Configuración de pytest para permitir importar módulos desde la carpeta src.

Este archivo agrega la raíz del proyecto al sys.path para que las pruebas
puedan importar correctamente módulos como src.dataset, src.perceptron,
src.evaluation y src.visualization.
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
