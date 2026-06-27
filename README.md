# Perceptrón de Frank Rosenblatt para selección preliminar de candidatos

Proyecto académico desarrollado en Python para analizar, implementar y probar el **perceptrón de Frank Rosenblatt** aplicado a una clasificación binaria: determinar si un candidato pasa o no a entrevista laboral a partir de dos criterios mínimos.

## Enlaces del proyecto

- **Repositorio GitHub:** https://github.com/antoniot73/perceptron-candidatos-cv
- **GitHub Pages:** https://antoniot73.github.io/perceptron-candidatos-cv/
- **Binder:** https://mybinder.org/v2/gh/antoniot73/perceptron-candidatos-cv/main?filepath=notebooks/practica_perceptron_candidatos.ipynb

## Objetivo

Implementar una versión propia del perceptrón de Frank Rosenblatt para observar el proceso completo de aprendizaje supervisado: carga y validación de datos, prueba sin entrenamiento, entrenamiento, evaluación, visualización y prueba con nuevos candidatos.

## Criterios del problema

El modelo utiliza dos entradas binarias derivadas de variables normalizadas:

- `cumple_experiencia`: 1 si la experiencia laboral normalizada es mayor o igual a 0.70.
- `cumple_puntaje`: 1 si el puntaje técnico normalizado es mayor o igual a 0.70.

La salida esperada es:

- `1`: pasa a entrevista.
- `0`: no pasa a entrevista.

La regla equivale a una lógica AND: el candidato pasa solo si cumple simultáneamente experiencia mínima y puntaje técnico mínimo.

## Tecnologías utilizadas

- Python
- NumPy
- Pandas
- Matplotlib
- Jupyter Notebook
- Pytest
- GitHub Pages
- Binder

## Estructura del proyecto

```text
perceptron_candidatos/
│
├── data/
│   ├── candidatos_30.csv
│   └── candidatos_prueba.csv
│
├── docs/
│   ├── index.html
│   └── practica_perceptron_candidatos.html
│
├── notebooks/
│   └── practica_perceptron_candidatos.ipynb
│
├── outputs/
│   ├── graficas/
│   └── tablas/
│
├── src/
│   ├── dataset.py
│   ├── evaluation.py
│   ├── perceptron.py
│   └── visualization.py
│
├── tests/
│   └── test_perceptron.py
│
├── main.py
├── requirements.txt
└── README.md
```

## Ejecución local

```bash
git clone https://github.com/antoniot73/perceptron-candidatos-cv.git
cd perceptron-candidatos-cv
python -m pip install -r requirements.txt
python main.py
pytest -v
```

## Generar HTML

```bash
jupyter nbconvert --to html notebooks/practica_perceptron_candidatos.ipynb --output-dir docs
```

En PowerShell, para actualizar GitHub Pages:

```powershell
copy docs\practica_perceptron_candidatos.html docs\index.html
```

## Salidas generadas

- Tablas CSV en `outputs/tablas/`.
- Gráficas PNG en `outputs/graficas/`.
- HTML del notebook en `docs/`.

## Documentación técnica

El pipeline está documentado mediante docstrings en los módulos principales:

- `src/dataset.py`: carga, validación y preparación de datasets.
- `src/perceptron.py`: implementación orientada a objetos del perceptrón.
- `src/evaluation.py`: métricas, tablas y persistencia de resultados.
- `src/visualization.py`: generación de gráficas.
- `main.py`: orquestación del pipeline completo.

## Alcance académico

Este proyecto tiene finalidad didáctica. No constituye un sistema real de contratación ni debe utilizarse para tomar decisiones laborales automatizadas. Su propósito es explicar de forma reproducible el funcionamiento básico del perceptrón como unidad binaria fundamental del aprendizaje automático.

## Autor

**Antonio Nicolás Toro González**  
Maestría en Inteligencia Artificial para la Transformación Digital  
Instituto Internacional de Aguascalientes
