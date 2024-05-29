from entrenamiento import entrenamiento_ddpg, generar_datos_GNN, entrenamiento_gnn
from evaluacion import inicio_evaluacion, evaluacion_libre


if __name__ == "__main__":

    inicio = input("Seleccione el modo (Entrenamiento : 'en',  Evaluación : 'ev', Evaluación libre: 'ev-l'):")

    if(inicio == 'en'):
        entrenar_ddpg = input("Desea utilizar utilizar el modulo de entrenamiento de una DDPG desde cero? (Si/No): ")
        if(entrenar_ddpg == "Si"):
            entrenamiento_ddpg()
        generar_datos_GNN()
        print("Se han generado los datos de entrenamiento en la ubicación data/...")
        print("Procediendo a entrenar el modelo GNN....")
        entrenamiento_gnn()
        print("El modelo GNN está entrenado y la red guardada con sus parámetros en la ubicación 'parametros/GNN/'.")
        print("Session terminada...")
    elif(inicio == 'ev'):
        inicio_evaluacion()
        print("Session terminada...")
    elif(inicio == 'ev-l'):
        modelo = input("Seleccione una versión del modelo ('voraz', 'ddpg'): ")
        evaluacion_libre(version_modelo=modelo)
        print("Session terminada...")
        print("Se ha generado un .gif con una representación animada de la simulacion realizada.")
        del modelo
    else:
        print("Session terminada...")
    del inicio
    