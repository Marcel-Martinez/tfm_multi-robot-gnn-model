'''En este modulo se encuentra la funcion de generación de escenarios
en el cual se generan los escenarios de entrada para el algoritmo centralizado/distribuido, estos serán las posiciones
iniciales de todos los drones, objetivos y estación de carga (se asume que todos los drones empiezan con 100% de batería y desde la estación de carga
, en la misma posición por conveniencia)'''
import numpy as np
import math
from random import randrange
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Funciones de comportamiento de la bateria
'''
Versión 1 usando la función lineal de descarga de la batería (fuente: dji)

def descarga_bateria(estado_actual):
    t = (100 - estado_actual)*(55/100)
    return round(-(100/55)*(t+1) + 100)

def carga_bateria(estado_actual):
    return round(estado_actual+(100/30))

    '''

'''Versión 2 usando una cadena de Markov'''


def descarga_bateria(estado_actual):
    if(random.random()>0.85 and estado_actual>0):
        return estado_actual-1
    else:
        return estado_actual

def carga_bateria(estado_actual):
    if(random.random()>0.05 and estado_actual<10):
        return estado_actual+1
    else:
        return estado_actual


#dim dimensión nxn de la grid, las estaciónes de carga estarán en las posiciónes (0,0) y (5,5)

def estado_inicio(dim, n_robots, n_obj):
    estado = {}
    pos_robots = []
    for i in range(0, n_robots):
        if(random.random()>0.5):
            pos_robots.append([0, randrange(0, dim-1)])
        else:
            pos_robots.append([randrange(0, dim-1), 0])
    pos_robots = np.array(pos_robots, dtype='float32')
    #bateria = np.ones(n_robots)*100
    bateria = np.ones(n_robots)*10
    pos_obj = []

    for i in range(0, n_obj):
        pos_obj.append([randrange(1, dim-1), randrange(1, dim-1)])
    pos_obj = np.array(pos_obj, dtype='float32')

    estado['pos_robots'] = pos_robots
    estado['bateria'] = bateria
    estado['pos_obj'] = pos_obj

    return estado


class Estado:

    # constructor
    def __init__(self, pos_robots, bateria, pos_obj):
        self.pos_robots = pos_robots
        self.bateria = bateria
        self.pos_obj = pos_obj
        self.registro = []
        self.it = 0
        self.fov_range = 4
        self.obj_ya_cubierto = np.ones(len(pos_robots))*-1

    def reset_env(self, pos_robots, bateria, pos_obj):
        self.pos_robots = pos_robots
        self.bateria = bateria
        self.pos_obj = pos_obj

    def get_estado(self):
        estado = {}
        estado = (map(tuple,self.pos_robots.tolist()), tuple(self.bateria), map(tuple, self.pos_obj.tolist()), tuple(self.planificador()))
        estado = tuple(map(tuple, estado))

        return estado

    def get_estado_array(self):
        #return np.array([np.concatenate([self.pos_robots.flatten(), self.bateria.flatten(), self.pos_obj.flatten(), self.planificador().flatten()])])
        # claro porque el no sabe donde estan los obj, se guia por la info que obtiene del FOV de cada robot (usando el planificador)
        return np.array([np.concatenate([self.pos_robots.flatten(), self.bateria.flatten(), self.planificador().flatten()])])

    def get_estado_experto(self):
        return [self.pos_robots, self.bateria.flatten(), self.planificador().flatten(), self.obj_ya_cubierto]


    def planificador(self):
      #planificacion = np.random.randint(0,6, size=len(self.pos_robots))
      planificacion = np.ones(len(self.pos_robots))*40 # un numero alejado de sus opciones lo interpretarán como fuera del alcance de cualquier objetivo
      direcciones = {
          (0,1): 0,
          (1,1): 1,
          (1,-1): 2,
          (0,-1): 3,
          (-1,-1): 4,
          (-1,1): 5,
          (1,0): 2,
          (-1,0): 4
      }
      for i in range(0, len(self.pos_robots)):
        for j in range(0, len(self.pos_obj)):
            if(math.sqrt((self.pos_robots[i,0]-self.pos_obj[j,0])**2 + (self.pos_robots[i,1]-self.pos_obj[j,1])**2)<=self.fov_range):
              dir = (self.pos_obj[j] - self.pos_robots[i])/abs(self.pos_obj[j] - self.pos_robots[i])
              x = np.nan_to_num(dir)
              if(tuple(x) in direcciones):
                planificacion[i] = direcciones[tuple(x)]
      return planificacion


    def verif_accion(self, robot, accion):
        nueva_pos = None
        acciones_contrarias = {0:3, 1:4, 2:5, 3:0, 4:1, 5:2}
        acciones_esquinas = {(0, 0): 1, (30, 0): 5, (30, 30): 4, (0, 30): 2}
        if(accion==0):
            nueva_pos = [self.pos_robots[robot,0], self.pos_robots[robot,1]+1]
        elif(accion==1):
            nueva_pos = [self.pos_robots[robot,0]+1, self.pos_robots[robot,1]+0.5]
        elif(accion==2):
            nueva_pos = [self.pos_robots[robot,0]+1, self.pos_robots[robot,1]-0.5]
        elif(accion==3):
            nueva_pos = [self.pos_robots[robot,0], self.pos_robots[robot,1]-1]
        elif(accion==4):
            nueva_pos = [self.pos_robots[robot,0]-1, self.pos_robots[robot,1]-0.5]
        elif(accion==5):
            nueva_pos = [self.pos_robots[robot,0]-1, self.pos_robots[robot,1]+0.5]

        if(nueva_pos == None):
          return accion
        elif(tuple(nueva_pos) in acciones_esquinas):
          return acciones_esquinas[tuple(nueva_pos)]
        elif(nueva_pos[0]>30 or nueva_pos[0]<0 or nueva_pos[1]>30 or nueva_pos[1]<0):
          return acciones_contrarias[accion]
        else:
          return accion



    def step(self, accion):
        # actualizar posicion de todos los i robots
        for i in range(0, len(self.pos_robots)):

            if(self.bateria[i]<=0):
                continue

            accion[i] = self.verif_accion(i, accion[i]) #correcion de acciones, en caso de que se intente salir del limite del mapa

            if(accion[i]==0):
                self.pos_robots[i] = [self.pos_robots[i,0], self.pos_robots[i,1]+1]
            elif(accion[i]==1):
                self.pos_robots[i] = [self.pos_robots[i,0]+1, self.pos_robots[i,1]+0.5]
            elif(accion[i]==2):
                self.pos_robots[i] = [self.pos_robots[i,0]+1, self.pos_robots[i,1]-0.5]
            elif(accion[i]==3):
                self.pos_robots[i] = [self.pos_robots[i,0], self.pos_robots[i,1]-1]
            elif(accion[i]==4):
                self.pos_robots[i] = [self.pos_robots[i,0]-1, self.pos_robots[i,1]-0.5]
            elif(accion[i]==5):
                self.pos_robots[i] = [self.pos_robots[i,0]-1, self.pos_robots[i,1]+0.5]
            else:
                pass # la accion 6 es que elige mantenerse en posicion

            # actualizar el estado de la bateria
            if((self.pos_robots[i]==[0,0]).all() or (self.pos_robots[i]==[15,15]).all()):
                self.bateria[i] = carga_bateria(self.bateria[i])
            else:
                self.bateria[i] = descarga_bateria(self.bateria[i])

            #actualizar pos objetivos
        for i in range(0, len(self.pos_obj)):
            dir = np.random.choice([0,1,2,3,4,5,6], p=[0, 0.15, 0.1, 0, 0.05, 0, 0.7])
            if(dir==0 and self.pos_obj[i,0]<30 and self.pos_obj[i,1]+1<30):
                self.pos_obj[i] = [self.pos_obj[i,0], self.pos_obj[i,1]+1]
            elif(dir==1 and self.pos_obj[i,0]+1<30 and self.pos_obj[i,1]+0.5<30):
                self.pos_obj[i] = [self.pos_obj[i,0]+1, self.pos_obj[i,1]+0.5]
            elif(dir==2 and self.pos_obj[i,0]+1<30 and self.pos_obj[i,1]-0.5>=0):
                self.pos_obj[i] = [self.pos_obj[i,0]+1, self.pos_obj[i,1]-0.5]
            elif(dir==3 and self.pos_obj[i,0]<30 and self.pos_obj[i,1]-1>=0):
                self.pos_obj[i] = [self.pos_obj[i,0], self.pos_obj[i,1]-1]
            elif(dir==4 and self.pos_obj[i,0]-1>=0 and self.pos_obj[i,1]-0.5>=0):
                self.pos_obj[i] = [self.pos_obj[i,0]-1, self.pos_obj[i,1]-0.5]
            elif(dir==5 and self.pos_obj[i,0]-1>=0 and self.pos_obj[i,1]+0.5<30):
                self.pos_obj[i] = [self.pos_obj[i,0]-1, self.pos_obj[i,1]+0.5]
            else:
                pass
        self.insertar_registro()

        #actualizamos el vector obj_ya_cubierto
        for i in range(0, len(self.pos_robots)):
            if(self.bateria[i]<=0):
                self.obj_ya_cubierto[i] = -1
                continue
            FOV1 = {}
            for k in [0, 0.5, 1]:
                for l in [0, 0.5, 1]:
                    FOV1[self.pos_robots[i,0]+k, self.pos_robots[i,1]] = 0
                    FOV1[self.pos_robots[i,0]+k, self.pos_robots[i,1]+l] = 0
                    FOV1[self.pos_robots[i,0], self.pos_robots[i,1]+l] = 0
                    FOV1[self.pos_robots[i,0]-k, self.pos_robots[i,1]+l] = 0
                    FOV1[self.pos_robots[i,0]-k, self.pos_robots[i,1]] = 0
                    FOV1[self.pos_robots[i,0]-k, self.pos_robots[i,1]-l] = 0
                    FOV1[self.pos_robots[i,0], self.pos_robots[i,1]-l] = 0
                    FOV1[self.pos_robots[i,0]+k, self.pos_robots[i,1]-l] = 0
                    FOV1[self.pos_robots[i,0], self.pos_robots[i,1]] = 0

            self.obj_ya_cubierto[i] = -1 # primero empezamos diciendo que no está cubriendo a nadie, y después comprobamos

            for j in range(0, len(self.pos_obj)):
                if(tuple(self.pos_obj[j]) in FOV1):
                    self.obj_ya_cubierto[i] = j



    def evaluar_posiciones(self):
        #para cada robot, se va a evaluar su posición utilizando el siguiente proceso:
        # recorriendo la matriz donde está la posición de cada robot:
            # para cada posición de robot (osea, para cada robot) generar el FOV (uno de radio 1)
             # con el de radio 1 verificar y contar la cantidad de obj cuyas posiciónes estan en él
             # meter en un vector en cada posición para cada robot la siguiente operación: 100*n_obj_FOV_1 - penalizacion
        #Retornar el vector resultante (que serán las recompensas del sistema en el estado actual tras evaluarse). Esto servirá para la funcion R(s,a) ya que esta es
        # literal la recompensa por el estado en el que se está
        vector_recompensa = np.zeros(len(self.pos_robots))

        obj_ya_cubierto_copia = self.obj_ya_cubierto.copy()

        for i in range(0, len(self.pos_robots)):
            if(self.bateria[i]<=0):
                continue
            penalizacion = 0
            FOV1 = {}
            for k in [0, 0.5, 1]:
                for l in [0, 0.5, 1]:
                    FOV1[self.pos_robots[i,0]+k, self.pos_robots[i,1]] = 0
                    FOV1[self.pos_robots[i,0]+k, self.pos_robots[i,1]+l] = 0
                    FOV1[self.pos_robots[i,0], self.pos_robots[i,1]+l] = 0
                    FOV1[self.pos_robots[i,0]-k, self.pos_robots[i,1]+l] = 0
                    FOV1[self.pos_robots[i,0]-k, self.pos_robots[i,1]] = 0
                    FOV1[self.pos_robots[i,0]-k, self.pos_robots[i,1]-l] = 0
                    FOV1[self.pos_robots[i,0], self.pos_robots[i,1]-l] = 0
                    FOV1[self.pos_robots[i,0]+k, self.pos_robots[i,1]-l] = 0
                    FOV1[self.pos_robots[i,0], self.pos_robots[i,1]] = 0

            obj_ya_cubierto_copia[i] = -1 # primero empezamos diciendo que no está cubriendo a nadie, y después comprobamos

            for j in range(0, len(self.pos_obj)):
                if(tuple(self.pos_obj[j]) in FOV1 and j not in obj_ya_cubierto_copia):
                    FOV1[tuple(self.pos_obj[j])] +=1
                    obj_ya_cubierto_copia[i] = j

            if((self.pos_robots[i]==[0,0]).all() or (self.pos_robots[i]==[15,15]).all()):
              penalizacion = -20
            else:
              penalizacion = 0

            vector_recompensa[i] = 100*sum(FOV1.values()) + penalizacion
        return sum(vector_recompensa)

    def insertar_registro(self):
      self.registro.append([self.it , self.pos_robots[:,0].copy(), self.pos_robots[:,1].copy(), self.pos_obj[:,0].copy(), self.pos_obj[:,1].copy()])
      self.it = self.it + 1

    def obtener_registro(self):
      return self.registro

    def generar_animacion(self, animation_name = "animacion.gif"):
      #obtenemos las ubicaciones de los robots y los objetivos
      robots = []
      for it in range(0, self.it):
        robots.append(pd.DataFrame([it, self.obtener_registro()[it][1], self.obtener_registro()[it][2]]).transpose())
      df_robots = pd.concat(robots)
      df_robots.columns = ['it', 'x0', 'y0']

      objetivos = []
      for it in range(0, self.it):
        objetivos.append(pd.DataFrame([self.obtener_registro()[it][0], self.obtener_registro()[it][3], self.obtener_registro()[it][4]]).transpose())
      df_objetivos = pd.concat(objetivos)
      df_objetivos.columns = ['it', 'x0', 'y0']

      fig, ax = plt.subplots()
      ax.set_xlim([0, 31])
      ax.set_ylim([0, 31])

      def animate(i):
        p1 = ax.clear()
        p2 = ax.clear()
        ax.grid(True)
        ax.set_xticks(np.linspace(0, 30, 31))
        ax.set_yticks(np.linspace(0, 30, 31))
        df_img1 = df_robots[df_robots['it'] == i][['x0', 'y0']]
        df_img2 = df_objetivos[df_objetivos['it'] == i][['x0', 'y0']]
        ax.set_xlim([0, 31])
        ax.set_ylim([0, 31])
        p1 = plt.scatter(df_img1['x0'].squeeze(), df_img1['y0'].squeeze())
        p2 = plt.scatter(df_img2['x0'].squeeze(), df_img2['y0'].squeeze())

        return p1,p2

      ani = animation.FuncAnimation(fig, func = animate, frames = 7, interval = 400, blit = True)
      ani.save(animation_name, writer="Pillow")


### Clases y Funciones de comportamiento Determinista para evaluación ###

def descarga_bateria_det(estado_actual, semilla_step):
    if(semilla_step>0.85 and estado_actual>0):
        return estado_actual-1
    else:
        return estado_actual

def carga_bateria_det(estado_actual, semilla_step):
    if(semilla_step>0.05 and estado_actual<10):
        return estado_actual+1
    else:
        return estado_actual

#dim dimensión nxn de la grid, las estaciónes de carga estarán en las posiciónes (0,0) y (5,5)

def estado_inicio_det(dim, n_robots, semilla_pos_robots, n_obj, semilla_pos_obj):
    estado = {}
    pos_robots = []
    for i in range(0, n_robots):
        pos_robots.append(semilla_pos_robots[i])
    pos_robots = np.array(pos_robots, dtype='float32')
    #bateria = np.ones(n_robots)*100
    bateria = np.ones(n_robots)*10
    pos_obj = []

    for i in range(0, n_obj):
        pos_obj.append(semilla_pos_obj[i])
    pos_obj = np.array(pos_obj, dtype='float32')

    estado['pos_robots'] = pos_robots
    estado['bateria'] = bateria
    estado['pos_obj'] = pos_obj

    return estado


class Estado_Determinista:

    # constructor
    def __init__(self, pos_robots, bateria, pos_obj, semilla_mov_obj, semilla_baterias):
        self.pos_robots = pos_robots
        self.bateria = bateria
        self.pos_obj = pos_obj
        self.registro = []
        self.it = 0
        self.fov_range = 4
        self.obj_ya_cubierto = np.ones(len(pos_robots))*-1
        self.semilla_mov_obj = semilla_mov_obj
        self.semilla_baterias = semilla_baterias

    def reset_env(self, pos_robots, bateria, pos_obj):
        self.pos_robots = pos_robots
        self.bateria = bateria
        self.pos_obj = pos_obj

    def get_estado(self):
        estado = {}
        estado = (map(tuple,self.pos_robots.tolist()), tuple(self.bateria), map(tuple, self.pos_obj.tolist()), tuple(self.planificador()))
        estado = tuple(map(tuple, estado))

        return estado

    def get_estado_array(self):
        #return np.array([np.concatenate([self.pos_robots.flatten(), self.bateria.flatten(), self.pos_obj.flatten(), self.planificador().flatten()])])
        # claro porque el no sabe donde estan los obj, se guia por la info que obtiene del FOV de cada robot (usando el planificador)
        return np.array([np.concatenate([self.pos_robots.flatten(), self.bateria.flatten(), self.planificador().flatten()])])

    def get_estado_experto(self):
        return [self.pos_robots, self.bateria.flatten(), self.planificador().flatten(), self.obj_ya_cubierto]


    def planificador(self):
      #planificacion = np.random.randint(0,6, size=len(self.pos_robots))
      planificacion = np.ones(len(self.pos_robots))*40 # un numero alejado de sus opciones lo interpretarán como fuera del alcance de cualquier objetivo
      direcciones = {
          (0,1): 0,
          (1,1): 1,
          (1,-1): 2,
          (0,-1): 3,
          (-1,-1): 4,
          (-1,1): 5,
          (1,0): 2,
          (-1,0): 4
      }
      for i in range(0, len(self.pos_robots)):
        for j in range(0, len(self.pos_obj)):
            if(math.sqrt((self.pos_robots[i,0]-self.pos_obj[j,0])**2 + (self.pos_robots[i,1]-self.pos_obj[j,1])**2)<=self.fov_range):
              dir = (self.pos_obj[j] - self.pos_robots[i])/abs(self.pos_obj[j] - self.pos_robots[i])
              x = np.nan_to_num(dir)
              if(tuple(x) in direcciones):
                planificacion[i] = direcciones[tuple(x)]
      return planificacion


    def verif_accion(self, robot, accion):
        nueva_pos = None
        acciones_contrarias = {0:3, 1:4, 2:5, 3:0, 4:1, 5:2}
        acciones_esquinas = {(0, 0): 1, (30, 0): 5, (30, 30): 4, (0, 30): 2}
        if(accion==0):
            nueva_pos = [self.pos_robots[robot,0], self.pos_robots[robot,1]+1]
        elif(accion==1):
            nueva_pos = [self.pos_robots[robot,0]+1, self.pos_robots[robot,1]+0.5]
        elif(accion==2):
            nueva_pos = [self.pos_robots[robot,0]+1, self.pos_robots[robot,1]-0.5]
        elif(accion==3):
            nueva_pos = [self.pos_robots[robot,0], self.pos_robots[robot,1]-1]
        elif(accion==4):
            nueva_pos = [self.pos_robots[robot,0]-1, self.pos_robots[robot,1]-0.5]
        elif(accion==5):
            nueva_pos = [self.pos_robots[robot,0]-1, self.pos_robots[robot,1]+0.5]

        if(nueva_pos == None):
          return accion
        elif(tuple(nueva_pos) in acciones_esquinas):
          return acciones_esquinas[tuple(nueva_pos)]
        elif(nueva_pos[0]>30 or nueva_pos[0]<0 or nueva_pos[1]>30 or nueva_pos[1]<0):
          return acciones_contrarias[accion]
        else:
          return accion



    def step(self, accion, sim_step):
        # actualizar posicion de todos los i robots
        for i in range(0, len(self.pos_robots)):

            if(self.bateria[i]<=0):
                continue

            accion[i] = self.verif_accion(i, accion[i]) #correcion de acciones, en caso de que se intente salir del limite del mapa

            if(accion[i]==0):
                self.pos_robots[i] = [self.pos_robots[i,0], self.pos_robots[i,1]+1]
            elif(accion[i]==1):
                self.pos_robots[i] = [self.pos_robots[i,0]+1, self.pos_robots[i,1]+0.5]
            elif(accion[i]==2):
                self.pos_robots[i] = [self.pos_robots[i,0]+1, self.pos_robots[i,1]-0.5]
            elif(accion[i]==3):
                self.pos_robots[i] = [self.pos_robots[i,0], self.pos_robots[i,1]-1]
            elif(accion[i]==4):
                self.pos_robots[i] = [self.pos_robots[i,0]-1, self.pos_robots[i,1]-0.5]
            elif(accion[i]==5):
                self.pos_robots[i] = [self.pos_robots[i,0]-1, self.pos_robots[i,1]+0.5]
            else:
                pass # la accion 6 es que elige mantenerse en posicion

            # actualizar el estado de la bateria
            if((self.pos_robots[i]==[0,0]).all() or (self.pos_robots[i]==[15,15]).all()):
                self.bateria[i] = carga_bateria_det(self.bateria[i], self.semilla_baterias[sim_step])
            else:
                self.bateria[i] = descarga_bateria_det(self.bateria[i], self.semilla_baterias[sim_step])

            #actualizar pos objetivos
        for i in range(0, len(self.pos_obj)):
            dir = self.semilla_mov_obj[sim_step, i]
            if(dir==0 and self.pos_obj[i,0]<30 and self.pos_obj[i,1]+1<30):
                self.pos_obj[i] = [self.pos_obj[i,0], self.pos_obj[i,1]+1]
            elif(dir==1 and self.pos_obj[i,0]+1<30 and self.pos_obj[i,1]+0.5<30):
                self.pos_obj[i] = [self.pos_obj[i,0]+1, self.pos_obj[i,1]+0.5]
            elif(dir==2 and self.pos_obj[i,0]+1<30 and self.pos_obj[i,1]-0.5>=0):
                self.pos_obj[i] = [self.pos_obj[i,0]+1, self.pos_obj[i,1]-0.5]
            elif(dir==3 and self.pos_obj[i,0]<30 and self.pos_obj[i,1]-1>=0):
                self.pos_obj[i] = [self.pos_obj[i,0], self.pos_obj[i,1]-1]
            elif(dir==4 and self.pos_obj[i,0]-1>=0 and self.pos_obj[i,1]-0.5>=0):
                self.pos_obj[i] = [self.pos_obj[i,0]-1, self.pos_obj[i,1]-0.5]
            elif(dir==5 and self.pos_obj[i,0]-1>=0 and self.pos_obj[i,1]+0.5<30):
                self.pos_obj[i] = [self.pos_obj[i,0]-1, self.pos_obj[i,1]+0.5]
            else:
                pass
        self.insertar_registro()

        #actualizamos el vector obj_ya_cubierto
        for i in range(0, len(self.pos_robots)):
            if(self.bateria[i]<=0):
                self.obj_ya_cubierto[i] = -1
                continue
            FOV1 = {}
            for k in [0, 0.5, 1]:
                for l in [0, 0.5, 1]:
                    FOV1[self.pos_robots[i,0]+k, self.pos_robots[i,1]] = 0
                    FOV1[self.pos_robots[i,0]+k, self.pos_robots[i,1]+l] = 0
                    FOV1[self.pos_robots[i,0], self.pos_robots[i,1]+l] = 0
                    FOV1[self.pos_robots[i,0]-k, self.pos_robots[i,1]+l] = 0
                    FOV1[self.pos_robots[i,0]-k, self.pos_robots[i,1]] = 0
                    FOV1[self.pos_robots[i,0]-k, self.pos_robots[i,1]-l] = 0
                    FOV1[self.pos_robots[i,0], self.pos_robots[i,1]-l] = 0
                    FOV1[self.pos_robots[i,0]+k, self.pos_robots[i,1]-l] = 0
                    FOV1[self.pos_robots[i,0], self.pos_robots[i,1]] = 0

            self.obj_ya_cubierto[i] = -1 # primero empezamos diciendo que no está cubriendo a nadie, y después comprobamos

            for j in range(0, len(self.pos_obj)):
                if(tuple(self.pos_obj[j]) in FOV1):
                    self.obj_ya_cubierto[i] = j



    def evaluar_posiciones(self):
        #para cada robot, se va a evaluar su posición utilizando el siguiente proceso:
        # recorriendo la matriz donde está la posición de cada robot:
            # para cada posición de robot (osea, para cada robot) generar el FOV (uno de radio 1)
             # con el de radio 1 verificar y contar la cantidad de obj cuyas posiciónes estan en él
             # meter en un vector en cada posición para cada robot la siguiente operación: 100*n_obj_FOV_1 - penalizacion
        #Retornar el vector resultante (que serán las recompensas del sistema en el estado actual tras evaluarse). Esto servirá para la funcion R(s,a) ya que esta es
        # literal la recompensa por el estado en el que se está
        vector_recompensa = np.zeros(len(self.pos_robots))

        for i in range(0, len(self.pos_robots)):
            if(self.bateria[i]<=0):
                continue
            penalizacion = 0
            FOV1 = {}
            for k in [0, 0.5, 1]:
                for l in [0, 0.5, 1]:
                    FOV1[self.pos_robots[i,0]+k, self.pos_robots[i,1]] = 0
                    FOV1[self.pos_robots[i,0]+k, self.pos_robots[i,1]+l] = 0
                    FOV1[self.pos_robots[i,0], self.pos_robots[i,1]+l] = 0
                    FOV1[self.pos_robots[i,0]-k, self.pos_robots[i,1]+l] = 0
                    FOV1[self.pos_robots[i,0]-k, self.pos_robots[i,1]] = 0
                    FOV1[self.pos_robots[i,0]-k, self.pos_robots[i,1]-l] = 0
                    FOV1[self.pos_robots[i,0], self.pos_robots[i,1]-l] = 0
                    FOV1[self.pos_robots[i,0]+k, self.pos_robots[i,1]-l] = 0
                    FOV1[self.pos_robots[i,0], self.pos_robots[i,1]] = 0

            self.obj_ya_cubierto[i] = -1 # primero empezamos diciendo que no está cubriendo a nadie, y después comprobamos

            for j in range(0, len(self.pos_obj)):
                if(tuple(self.pos_obj[j]) in FOV1 and j not in self.obj_ya_cubierto):
                    FOV1[tuple(self.pos_obj[j])] +=1
                    self.obj_ya_cubierto[i] = j

            if((self.pos_robots[i]==[0,0]).all() or (self.pos_robots[i]==[15,15]).all()):
              penalizacion = -20
            else:
              penalizacion = 0

            vector_recompensa[i] = 100*sum(FOV1.values()) + penalizacion
        return sum(vector_recompensa)

    def insertar_registro(self):
      self.registro.append([self.it , self.pos_robots[:,0].copy(), self.pos_robots[:,1].copy(), self.pos_obj[:,0].copy(), self.pos_obj[:,1].copy()])
      self.it = self.it + 1

    def obtener_registro(self):
      return self.registro

    def generar_animacion(self, animation_name = "animacion.gif"):
      #obtenemos las ubicaciones de los robots y los objetivos
      robots = []
      for it in range(0, self.it):
        robots.append(pd.DataFrame([it, self.obtener_registro()[it][1], self.obtener_registro()[it][2]]).transpose())
      df_robots = pd.concat(robots)
      df_robots.columns = ['it', 'x0', 'y0']

      objetivos = []
      for it in range(0, self.it):
        objetivos.append(pd.DataFrame([self.obtener_registro()[it][0], self.obtener_registro()[it][3], self.obtener_registro()[it][4]]).transpose())
      df_objetivos = pd.concat(objetivos)
      df_objetivos.columns = ['it', 'x0', 'y0']

      fig, ax = plt.subplots()
      ax.set_xlim([0, 31])
      ax.set_ylim([0, 31])

      def animate(i):
        p1 = ax.clear()
        p2 = ax.clear()
        ax.grid(True)
        ax.set_xticks(np.linspace(0, 30, 31))
        ax.set_yticks(np.linspace(0, 30, 31))
        df_img1 = df_robots[df_robots['it'] == i][['x0', 'y0']]
        df_img2 = df_objetivos[df_objetivos['it'] == i][['x0', 'y0']]
        ax.set_xlim([0, 31])
        ax.set_ylim([0, 31])
        p1 = plt.scatter(df_img1['x0'].squeeze(), df_img1['y0'].squeeze())
        p2 = plt.scatter(df_img2['x0'].squeeze(), df_img2['y0'].squeeze())

        return p1,p2

      ani = animation.FuncAnimation(fig, func = animate, frames = self.it, interval = 400, blit = True)
      ani.save(animation_name, writer="Pillow")