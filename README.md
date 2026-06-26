# Práctica: análisis, implementación y prueba del algoritmo del perceptrón de Frank Rosenblatt aplicado a la selección preliminar de candidatos para entrevista laboral

<div align="center">

**Instituto Internacional de Aguascalientes**  

**Maestría en Inteligencia Artificial para la Transformación Digital**  

**Programa:** Aprendizaje Inteligente  

<br>

**Alumno:** Antonio Nicolás Toro González  

**Tutor:** Dr. Francisco Javier Luna Rosas  

<br><br>

**Junio 2026**

</div>

---

## Objetivo

Analizar, implementar y probar el perceptrón de Frank Rosenblatt aplicado a un problema de selección de cumplimiento: decidir si un candidato pasa o no pasa a entrevista laboral.

El proyecto compara dos escenarios:

1. Perceptrón sin entrenamiento.
2. Perceptrón entrenado mediante la regla de aprendizaje de Rosenblatt.

Esta comparación permite observar que un perceptrón sin entrenamiento puede producir predicciones poco confiables, mientras que un perceptrón entrenado ajusta sus pesos y sesgo a partir del error para aprender una regla de clasificación binaria.

## Repositorio y ejecución en línea

Repositorio público del proyecto:

[https://github.com/antoniot73/perceptron-candidatos-cv](https://github.com/antoniot73/perceptron-candidatos-cv)

Puedes abrir y ejecutar la práctica en Binder desde el navegador:

[![Abrir en Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/antoniot73/perceptron-candidatos-cv/main?filepath=notebooks/practica_perceptron_candidatos.ipynb)

En Binder, espera a que termine la construcción del entorno y luego abre el notebook:

```text
notebooks/practica_perceptron_candidatos.ipynb
```

## Marco teórico incorporado

El notebook incluye un marco teórico sintetizado con dos apartados:

1. Conceptos generales de Machine Learning.
2. El perceptrón de Frank Rosenblatt.

Este marco teórico se ubica después de la portada y antes de la motivación del proyecto. Las referencias bibliográficas se mantienen al final de la práctica, después de las conclusiones y la reflexión final.

## Motivación de la práctica

**Práctica: análisis, implementación y prueba del algoritmo del perceptrón de Frank Rosenblatt aplicado a la selección preliminar de candidatos para entrevista laboral.**

Esta práctica usa el perceptrón para analizar un problema binario de cumplimiento: decidir si un candidato pasa o no pasa a entrevista.

El caso es académico y no pretende automatizar una contratación real. La clasificación se basa en dos criterios mínimos: experiencia laboral y puntaje técnico. Ambos deben cumplirse simultáneamente para obtener una salida positiva.

## Planteamiento del problema

Cada candidato se representa mediante dos variables normalizadas:

- `x1`: experiencia laboral normalizada.
- `x2`: puntaje técnico normalizado.

La experiencia laboral se mide en **años de experiencia, de 0 a 4 años o más**, y se normaliza así:

```text
x1 = años de experiencia / 4
```

Por tanto, `x1 = 1.00` equivale a 4 años o más, y `x1 = 0.70` equivale aproximadamente a 2.8 años.

El puntaje técnico se mide en **una evaluación de 0 a 100 puntos**, y se normaliza así:

```text
x2 = puntaje técnico / 100
```

Por tanto, `x2 = 1.00` equivale a 100 puntos, y `x2 = 0.70` equivale a 70 puntos.

| Variable | Medición original | Rango normalizado | Mínimo aprobatorio |
|---|---|---:|---:|
| `x1` | Años de experiencia, de 0 a 4 años o más | 0.00 a 1.00 | 0.70 = 2.8 años |
| `x2` | Puntaje técnico de 0 a 100 puntos | 0.00 a 1.00 | 0.70 = 70 puntos |

La salida esperada es binaria:

```text
1 -> pasa a entrevista
0 -> no pasa a entrevista
```

La regla de decisión es:

```text
si x1 >= 0.70 y x2 >= 0.70 -> y = 1
si x1 < 0.70 o x2 < 0.70  -> y = 0
```

Por ello, no basta con que una variable sea alta si la otra no alcanza el mínimo aprobatorio.

Para que el perceptrón simple aprenda esta regla tipo AND de forma consistente, las variables normalizadas se transforman en dos indicadores binarios de cumplimiento:

```text
c1 = 1 si x1 >= 0.70; si no, c1 = 0
c2 = 1 si x2 >= 0.70; si no, c2 = 0
```

Las entradas reales del perceptrón son `c1` y `c2`.

Las condiciones iniciales del perceptrón son controladas para reproducibilidad:

```text
w1 = 0.0
w2 = 0.0
b = 0.0
eta = 0.1
max_epochs = 100
```

## Variables

- `anios_experiencia`: experiencia laboral original.
- `puntaje_tecnico`: puntaje técnico original.
- `x1_experiencia_norm`: experiencia laboral normalizada.
- `x2_puntaje_norm`: puntaje técnico normalizado.
- `cumple_experiencia`: indicador binario de cumplimiento de experiencia mínima.
- `cumple_puntaje`: indicador binario de cumplimiento de puntaje mínimo.
- `suma_cumplimiento`: suma informativa de los criterios normalizados.
- `y_pasa_entrevista`: salida esperada para el dataset de entrenamiento.
- `y_esperada`: salida esperada para candidatos de prueba.

## Estructura del proyecto

```text
perceptron_candidatos/
├── data/
│   ├── candidatos_30.csv
│   └── candidatos_prueba.csv
├── notebooks/
│   └── practica_perceptron_candidatos.ipynb
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── perceptron.py
│   ├── evaluation.py
│   └── visualization.py
├── outputs/
│   ├── graficas/
│   ├── tablas/
│   └── reportes/
├── tests/
│   ├── conftest.py
│   └── test_perceptron.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Instalación local

Si se usa Binder, no es necesario instalar dependencias localmente. Para ejecutar el proyecto en una computadora personal, desde la carpeta raíz del proyecto:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En macOS/Linux:

```bash
source .venv/bin/activate
```

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución por consola

```bash
python main.py
```

Este comando ejecuta el flujo completo:

```text
carga de datos
validación
prueba sin entrenamiento
entrenamiento
prueba con entrenamiento
prueba con candidatos nuevos
generación de tablas
generación de gráficas
```

## Pruebas unitarias

```bash
pytest -v
```

## Ejecución en VS Code / Jupyter

Abrir:

```text
notebooks/practica_perceptron_candidatos.ipynb
```

Seleccionar el kernel de Python y ejecutar todas las celdas con **Run All**.

## Salidas esperadas

Tablas:

```text
outputs/tablas/resultados_sin_entrenamiento.csv
outputs/tablas/resultados_con_entrenamiento.csv
outputs/tablas/resultados_candidatos_prueba.csv
```

Gráficas:

```text
outputs/graficas/errores_por_epoca.png
outputs/graficas/criterios_minimos_aprobacion.png
outputs/graficas/frontera_and_aprendida.png
outputs/graficas/comparacion_exactitud.png
```

## Consideración ética

El proyecto es una simulación académica. No debe interpretarse como un sistema real de contratación ni sustituir la revisión humana. En una aplicación real deberían considerarse criterios legales, protección de datos personales, sesgos, transparencia, explicabilidad, auditoría y supervisión humana.


## Reflexión final

El perceptrón de Frank Rosenblatt es una unidad binaria básica del aprendizaje automático: recibe entradas, calcula una combinación ponderada y produce una decisión `0` o `1`. Su importancia radica en mostrar de forma simple el ciclo de aprendizaje supervisado: predicción, error y ajuste de pesos.

## Referencias Bibliográficas

Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep learning*. MIT Press.

Rosenblatt, F. (1958). The perceptron: A probabilistic model for information storage and organization in the brain. *Psychological Review, 65*(6), 386–408.

Russell, S. J., & Norvig, P. (2022). *Artificial intelligence: A modern approach* (4th ed., Global ed.). Pearson.


## Recursos EdTech de Machine Learning

1. [Perceptrón de Frank Rosenblatt Interactivo](https://perceptronfrankrosenblatt.streamlit.app/)  
   Nota de acceso: clicar **Yes, get this app back up!**

2. [Explorador interactivo de Machine Learning](https://antoniot73.shinyapps.io/ml-tipos-aprendizaje/)  
   Nota de acceso: clicar **Disconnected from the server. Reload**
