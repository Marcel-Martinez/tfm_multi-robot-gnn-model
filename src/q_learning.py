'''Definición del sistema experto para generación de los datos de entrenamiento del sistema de inteligencia distribuida.
En él, se definirá el MDP el algoritmo de RL que lo resolverá (Q-Learning) y finalmente una fución para generar (en un txt, csv....) 
el conjutno de datos de entramiento que serán simplemente el conjunto de estados explorados total y la matriz de acciones elegidas para el conjunto 
según el estado. Este será preprocesado y transformado por el gnn_data_setup
'''
from simulador_entorno import Estado, estado_inicio
import numpy as np
import random
import itertools

GAMMA = 0.8
SIMULACIONES = 100
Q = {}
DIMMENSION = 10 # nxn
n_robots = 5
n_obj = 3
ratio_exploracion = 0.1
ratio_aprendizaje = 0.01

#set de acciones
A = []
for i in itertools.product([0,1,2,3,4,5,6], repeat=n_robots):
    A.append(tuple(i))
#A tiene 16805 filas

def e_voraz(estado, ratio_exploracion):
    if(random.random() < ratio_exploracion):
        return np.random.randint(0, 7, size=n_robots)
    else:
        return mejor_accion(Q, estado)
    
def mejor_accion(Q, estado):
    mejor_q = 0
    mejor_a = []

    for i in A:
        res = Q_eval(Q, estado, i)
        if(res>=mejor_q):
            mejor_q = res
            mejor_a = i

    return mejor_a


def Q_eval(Q, estado, accion):
    accion = tuple(accion)

    if((estado, accion) not in Q):
        Q[estado, accion] = random.random()
        return Q[estado, accion]
    else:
        return Q[estado, accion]


# Q-Learning

def Q_Learning(Q, estado_inicial):

    estado = Estado(estado_inicial['pos_robots'].copy(), estado_inicial['bateria'].copy(), estado_inicial['pos_obj'].copy())

    for i in range(0, SIMULACIONES):
        estado.reset_env(estado_inicial['pos_robots'].copy(), estado_inicial['bateria'].copy(), estado_inicial['pos_obj'].copy())
        #print(Q)
        for j in range(0, 50):
# cada simulación va a dar 50 pasos
            estado_0 = estado.get_estado()
            accion = tuple(e_voraz(estado_0, ratio_exploracion))
            estado.step(accion)
            estado_1 = estado.get_estado()
            R = estado.evaluar_posiciones()
            #print("Vuelta: ",j)
            #print(accion)
 
            Q[estado_0, accion] = Q_eval(Q,estado_0, accion) + ratio_aprendizaje*(R + GAMMA*Q_eval(Q, estado_1, mejor_accion(Q, estado_1)) - Q_eval(Q, estado_0, accion))
    
    return Q
