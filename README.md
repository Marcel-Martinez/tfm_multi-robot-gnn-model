# tfm_multi-robot-gnn-model
Repositorio del código fuente de mi proyecto de fin de Master Universitario en Inteligencia Artificial de la Universidad de Sevilla. 

En este proyecto se resuelve un problema de búsqueda y seguimiento de objetivos móviles empleando agentes inteligentes operados por un modelo inteligencia artificial distribuida basada en Graph Neural Networks. Los agentes son representados por drones volando a distinta altura y los objetivos a vehículos terrestres.

<p align="center">
<img src="/Imagen1.jpg"  alt="drawing" width="500"/>
</p>

En este repositorio encontrará el notebook utilizado para investigación y desarrollo del proyecto y el marco de trabajo desarrollado con el cuál puede modelar y entrenar una red de agentes inteligentes empleando GNN y correr simulaciones de búsqueda y seguimiento de objetivos.

## Setup
Tras clonar el repositorio utilizando *git clone* deberá instalar las dependencias necesarias en requirement.txt.

### Requirements
```
tensorflow==2.15.0
tensorflow-gnn==1.0.2
numpy==1.24.3
scipy==1.11.1
pandas==2.0.3
matplotlib==3.7.2
```


## Entrenamiento
El proceso consiste en utilizar un agente que resuelve el problema de tracking de forma centralizada y efectiva para generar un conjunto de datos de entrenamiento para el modelo distribuido GNN.

Este proyecto dispone de dos soluciones centralizadas, una utilizando una función voraz para resolver el problema y otra empleando un modelo de aprendizaje por refuerzo DDPG.

Al correr el archivo `main.py` se presenta la opción de elegir el agente centralizado 'ddpg' o 'voraz'.

Si escoge el ‘ddpg’ como agente centralizado puede elegir cargar una arquitectura preentrenada o entrenar una desde cero en el entorno de simulación.

Tras ésto, el agente centralizado generará los datos de entrenamiento para la GNN y se procederá a entrenarse esta. Los datos del agente centralizado se guardarán en pickles en la ubicación `./data/voraz/` o `./data/DDPG`. Posteriormente se crea el dataset de entrenamiento conformado por objetos GraphTensors de la librería Tensorflow-GNN con la que se entrena el modelo. 

El dataset de entrenamiento se guarda en la ubicación `train_data/train_data.tfrecord` y los parámetros del modelo entrenado se guardan en `parametros/GNN/`

## Simulación

Puede correr el archivo `main.py` y elegir la acción de realizar una simulación de comparación con el agente centralizado o una simulación sola del modelo GNN entrenado (llamado “evaluación libre”).

Tras realizada una evaluación libre se genera un archivo .gif con la simulación representada como a continuación.

<p align="center">
<img src="/animacion_gnn.gif" alt="drawing" width="500"/>
</p>

Donde los puntos azules representan nuestros agentes inteligentes y los naranja los objetivos a seguir.
## Licencia
Este proyecto está bajo la licencia del MIT. Para mayores detalles véase el anexo de LICENCE.

