# definición del algoritmo experto a partir de una lógica voraz (humana/programada)
import random
import numpy as np
import itertools
'''estado=[baterias, LiDAR, objetivo_cubiertos]'''
def algoritmo_voraz(estado, n_robots):
    acciones_contrarias = {0:3, 1:4, 2:5, 3:0, 4:1, 5:2, 40:random.randint(0,5), 6:6}
    direcciones = {
        (0,1): 0,
        (1,1): 1,
        (1,-1): 2,
        (0,-1): 3,
        (-1,-1): 4,
        (-1,1): 5,
        (1,0): 2,
        (-1,0): 4,
        (0,0): 6
    }
    acciones = np.ones(n_robots)*6
    pos, baterias, LiDAR, objetivos_cubiertos = estado
    for i in range(0, n_robots):
        if(baterias[i]<=3):
            dir = ([15,15]-pos[i])/abs([15,15]-pos[i])
            x = np.nan_to_num(dir)
            acciones[i] = direcciones[tuple(x)] #dejar todo e ir a la estacion de carga
        else:
            if(objetivos_cubiertos[i] != -1 and objetivos_cubiertos[i] not in np.delete(objetivos_cubiertos, i)):
                acciones[i] = 6
            elif(objetivos_cubiertos[i] != -1 and objetivos_cubiertos[i] in np.delete(objetivos_cubiertos, i)):
                acciones[i] = acciones_contrarias[LiDAR[i]]
                objetivos_cubiertos[i] = -1
            elif(LiDAR[i] != 40):
                acciones[i] = LiDAR[i]
            else:
                acciones[i] = random.randint(0,5)

    return acciones

import random
import numpy as np
import itertools
import copy

def algoritmo_genetico(estado, poblacion_ini, generaciones = 15):
  n_robots = len(estado.get_estado()[0])
  poblacion = []
  #conjunto de acciones
  for i in itertools.product([0,1,2,3,4,5,6], repeat=n_robots):
      poblacion.append(list(i))
  poblacion = random.sample(poblacion,int((poblacion_ini/100)*len(poblacion)))

  for generacion in range(generaciones):
    #evaluar el fitness
    fitness_candidatos = []
    for val in poblacion:
      candidato = copy.copy(val)
      estado_aux = copy.copy(estado)
      estado_aux.step(candidato)
      candidato.append(estado_aux.evaluar_posiciones())

      fitness_candidatos.append(candidato)

    fitness_candidatos = np.array(fitness_candidatos)

    if(sum(fitness_candidatos[:, -1]) == 0):
      # si resulta que no hay ninguna solucion viable en la poblacion inicial, que pare ahí y devuelva random
      return np.random.randint(low=0, high=6, size=n_robots)

    #selecionamos los mejores
    fitness_candidatos = fitness_candidatos[np.argsort(fitness_candidatos[:, -1])]
    n_seleccion = int(len(fitness_candidatos)*0.1)
    padre1, padre2 = fitness_candidatos[:n_seleccion], fitness_candidatos[n_seleccion:n_seleccion*2]
    padre1, padre2 = padre1[:, :-1], padre2[:, :-1] #quitamos la columna del fitness

    #generamos hijos

    hijos = []
    for _ in range(10):
      if(len(padre1[0])%2 !=0):
        mitad = len(padre1[0])//2
      else:
        mitad = len(padre1[0])//2
      hijo = np.concatenate((padre1[:,:mitad], padre2[:,mitad:]), axis=1)
      hijos.append(hijo)
      np.random.shuffle(padre1)
    hijos = np.vstack(hijos)

    #mutacion (adjacent swap de dos genes)
    gen1, gen2 = np.random.choice(hijos.shape[1], 2, replace=False)
    hijos_mutados = np.copy(hijos)
    hijos_mutados[:, [gen1, gen2]] = hijos[:, [gen2, gen1]]

    poblacion = hijos_mutados.tolist()

  fitness_candidatos = []
  for val in poblacion:
    candidato = copy.copy(val)
    estado_aux = copy.copy(estado)
    estado_aux.step(candidato)
    candidato.append(estado_aux.evaluar_posiciones())

    fitness_candidatos.append(candidato)

  fitness_candidatos = np.array(fitness_candidatos)

  return fitness_candidatos[0]

