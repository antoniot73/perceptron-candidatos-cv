## 1. Marco teórico

### 1.1 Conceptos generales de Machine Learning

El aprendizaje de máquina, o *machine learning*, es una rama de la inteligencia artificial que estudia algoritmos capaces de mejorar su desempeño a partir de la experiencia. A diferencia de la programación tradicional, donde las reglas son definidas explícitamente por el programador, en *machine learning* el sistema aprende patrones desde datos y utiliza esos patrones para realizar predicciones, clasificaciones o decisiones.

Una definición clásica establece que un programa aprende de una experiencia **E**, respecto a una tarea **T** y una medida de desempeño **P**, si su desempeño en la tarea mejora con la experiencia. Esta idea permite entender el aprendizaje como un proceso medible: el sistema recibe datos, realiza una tarea y se evalúa mediante una métrica de desempeño.

Los antecedentes del aprendizaje de máquina se relacionan con la inteligencia artificial, la estadística, la cibernética y los primeros modelos inspirados en el sistema nervioso. Durante las décadas de 1940 a 1960 surgieron modelos neuronales artificiales que buscaban explicar cómo una máquina podía reconocer patrones y modificar su comportamiento. En este contexto, el perceptrón de Frank Rosenblatt, propuesto en 1958, se considera uno de los primeros modelos de aprendizaje automático y un antecedente directo de las redes neuronales artificiales.

### 1.2 El perceptrón de Frank Rosenblatt

El perceptrón es un modelo simple de clasificación binaria. Recibe entradas numéricas, les asigna pesos, calcula una suma ponderada y aplica una función de activación para producir una salida. De forma general, calcula:

$$
z = w \cdot x + b
$$

donde **x** representa el vector de entradas, **w** el vector de pesos y **b** el sesgo. Después aplica una función escalón:

$$
y =
\begin{cases}
1, & \text{si } z \geq 0 \\
0, & \text{si } z < 0
\end{cases}
$$

El aprendizaje ocurre cuando el modelo compara su predicción con la salida real y ajusta sus pesos si existe error:

$$
w_i = w_i + \eta (y - \hat{y})x_i
$$

El sesgo se actualiza mediante:

$$
b = b + \eta (y - \hat{y})
$$

donde **η** es la tasa de aprendizaje, **y** es la salida real y **ŷ** es la predicción del modelo.

La importancia del perceptrón radica en que muestra de forma clara cómo un modelo puede aprender una frontera de decisión a partir de ejemplos. Sin embargo, su principal limitación es que solo puede resolver problemas linealmente separables; por ello, no puede resolver correctamente problemas no lineales como XOR sin utilizar capas adicionales.

