'''Aquí se definirán las funciones/clases para el preprocesado y transformación de los datos generados por el
Sistema Experto y crear el conjunto de entrenamiento'''
import numpy as np
import math
FOV_RANGE = 4
COM_RANGE = 6

def generar_datos(estado, acciones):
    adj_mat = generar_adj_mat(estado[0], COM_RANGE) # calculada a partir de pos_robots (valor de la distancia incluido)
    FOV_mat, FOV_mat_pos = FOV_obj(estado[0], estado[2], estado[1], FOV_RANGE)
    atributos = generar_atributos_v2(adj_mat, FOV_mat, estado[1], estado[0], estado[3]) # lista que contiene la posicion (o distancia) y el numero de objetivos ¿y sus posiciones? de cada robot

    acciones = np.array(acciones)

    return atributos, adj_mat, acciones

def generar_adj_mat(pos_robots, com_range):
    pos = np.array(pos_robots)
    adj_mat = np.zeros((len(pos), len(pos)))

    for i in range(0, len(pos)):
        for j in range(0, len(pos)):
            #if(np.linalg.norm(pos[i] - pos[j]) == 0 and i!=j):
             #   adj_mat[i,j] = 1
            if(math.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2) <= com_range and i!=j):
                #adj_mat[i,j] = math.sqrt((pos[i,0]-pos[j,0])**2 + (pos[i,1]-pos[j,1])**2)
                adj_mat[i,j] = 1
            else:
                adj_mat[i,j] = 0

    return adj_mat



def generar_atributos(adj_mat, fov_obj, fov_vec_pos, bateria, pos_robots, planificador):
    # matriz de num_robots x (1 + 1 + 1 + num_robots)
    # el primer atributo es igual para todos, el estado de batería
    # el segundo 1 es el número de objetivos cubiertos
    # el tercer 1 es el número de objetivos en el FOV del robot (cubiertos o no)
    # el conjutno de columnas num_robots almacena para cada robot la distancia relativa entre cada robot en su FOV (es)
    bateria = np.array(bateria)
    #atributos = np.zeros((len(adj_mat), (len(adj_mat)+2))) version anterior del vector de atributos, ahora sólo se deja un numero con el numero de vecinos
    atributos = np.zeros((len(adj_mat), 13))
    #atributos = {}
    # para cada robot i
    for i in range(0, len(adj_mat)):

        atributos[i, 0] = bateria[i]

        atributos[i, 1] = (fov_obj[i,:] == 1).sum() # numero de objetivos cubiertos
        atributos[i, 2] = (fov_obj[i,:] >= 1).sum() # numero de objetivos en su FOV cubiertos o no
        atributos[i, 3] = pos_robots[i][0] #pos del robot
        atributos[i, 4] = pos_robots[i][1] #pos del robot
        min_pos = sorted(fov_vec_pos[i])[:3]
        for idx, val in enumerate(min_pos):
          atributos[i, idx*2+5] = val[0]
          atributos[i, idx*2+6] = val[1]
        #it = 0
        #for j in range(0, len(adj_mat)-1):
         #   if(it!=i):
                #agregar el vector de distancia relativa entre el dron y el resto dentro de su FOV
          #      atributos[i, j+3] = adj_mat[i,it]
           # elif(it==i):
            #    it +=1
             #   atributos[i, j+3] = adj_mat[i,it]
            #it +=1
        if(pos_robots[i] == [0,0] or pos_robots[i] == [15,15]):
            atributos[i, 11] = 1 #cargando
        atributos[i, 12] = planificador[i]

    return atributos

def generar_atributos_v2(adj_mat, fov_obj, bateria, pos_robots, planificador):
    # esta versión de la función generar_atributos es igual que la anterior pero se eliminan los atributos del conjunto de columnas que representaban vectores de distancia relativa calculada a cada vecino
    bateria = np.array(bateria)
    atributos = np.zeros((len(adj_mat), 7))

    for i in range(0, len(adj_mat)):

        atributos[i, 0] = bateria[i]

        atributos[i, 1] = (fov_obj[i,:] == 1).sum() # numero de objetivos cubiertos
        atributos[i, 2] = (fov_obj[i,:] >= 1).sum() # numero de objetivos en su FOV cubiertos o no
        atributos[i, 3] = pos_robots[i][0] #pos del robot
        atributos[i, 4] = pos_robots[i][1] #pos del robot

        if(pos_robots[i] == [0,0] or pos_robots[i] == [15,15]):
            atributos[i, 5] = 1 #cargando
        atributos[i, 6] = planificador[i]

    return atributos

def FOV_obj(pos_robots, pos_obj, bateria, FOV_range):
    robots = np.array(pos_robots)
    obj = np.array(pos_obj)
    FOV_mat = np.zeros((len(robots), len(obj))) # num_robot x num_obj


    for i in range(0, len(pos_robots)):

      if(bateria[i] == 0):
        continue

      FOV1 = {}
      for m in [0, 0.5, 1]:
        for l in [0, 0.5, 1]:
            FOV1[pos_robots[i][0], pos_robots[i][1]] = 0
            FOV1[pos_robots[i][0]+m, pos_robots[i][1]] = 0
            FOV1[pos_robots[i][0]+m, pos_robots[i][1]+l] = 0
            FOV1[pos_robots[i][0], pos_robots[i][1]+l] = 0
            FOV1[pos_robots[i][0]-m, pos_robots[i][1]+l] = 0
            FOV1[pos_robots[i][0]-m, pos_robots[i][1]] = 0
            FOV1[pos_robots[i][0]-m, pos_robots[i][1]-l] = 0
            FOV1[pos_robots[i][0], pos_robots[i][1]-l] = 0
            FOV1[pos_robots[i][0]+m, pos_robots[i][1]-l] = 0

      for j in range(0, len(pos_obj)):
          if(tuple(pos_obj[j]) in FOV1):
              FOV_mat[i,j] = 1 # 1 si el robot i tiene al obj j cubierto


    for i in range(0, len(pos_robots)):

      if(bateria[i] == 0):
        continue

      FOV2 = {}
      for k in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
        for l in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
            FOV2[pos_robots[i][0]+k, pos_robots[i][1]] = 0
            FOV2[pos_robots[i][0]+k, pos_robots[i][1]+l] = 0
            FOV2[pos_robots[i][0], pos_robots[i][1]+l] = 0
            FOV2[pos_robots[i][0]-k, pos_robots[i][1]+l] = 0
            FOV2[pos_robots[i][0]-k, pos_robots[i][1]] = 0
            FOV2[pos_robots[i][0]-k, pos_robots[i][1]-l] = 0
            FOV2[pos_robots[i][0], pos_robots[i][1]-l] = 0
            FOV2[pos_robots[i][0]+k, pos_robots[i][1]-l] = 0

      for j in range(0, len(pos_obj)):
        if(tuple(pos_obj[j]) in FOV2 and FOV_mat[i,j] == 1):
          FOV_mat[i,j] = 1
        elif(tuple(pos_obj[j]) in FOV2 and FOV_mat[i,j] != 1):
          FOV_mat[i,j] += 2 # 2 si el robot i tiene al obj j cubierto o en su fov

    FOV_mat_pos = {}
    for i in range(0, len(pos_robots)):
      lista = [[0,0]]
      FOV2 = {}
      FOV_mat_pos[i] = lista
      if(bateria[i] == 0):
        continue

      for k in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
        for l in [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]:
            FOV2[pos_robots[i][0]+k, pos_robots[i][1]] = 0
            FOV2[pos_robots[i][0]+k, pos_robots[i][1]+l] = 0
            FOV2[pos_robots[i][0], pos_robots[i][1]+l] = 0
            FOV2[pos_robots[i][0]-k, pos_robots[i][1]+l] = 0
            FOV2[pos_robots[i][0]-k, pos_robots[i][1]] = 0
            FOV2[pos_robots[i][0]-k, pos_robots[i][1]-l] = 0
            FOV2[pos_robots[i][0], pos_robots[i][1]-l] = 0
            FOV2[pos_robots[i][0]+k, pos_robots[i][1]-l] = 0

      #for j in range(0, len(pos_obj)):
       # if(tuple(pos_obj[j]) in FOV2):
        #  lista.append((np.array(pos_obj[j])-np.array(pos_robots[i])).tolist())
      #FOV_mat_pos[i] = lista
      for j in range(0, len(pos_robots)):
        if(i!=j and tuple(pos_robots[j]) in FOV2):
          lista.append((np.array(pos_robots[j])-np.array(pos_robots[i])).tolist())
      FOV_mat_pos[i] = lista

    return FOV_mat, FOV_mat_pos