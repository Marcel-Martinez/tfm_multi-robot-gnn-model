from simulador_entorno import Estado_Determinista, estado_inicio_det, carga_bateria_det, descarga_bateria_det
from gnn_data_setup import *
from ddpg import *
from greedys_algorithms import algoritmo_voraz
from gnn_model import *
import tensorflow as tf
import tensorflow_gnn as tfgnn
import pickle

def simulacion_DDPG(estado_inicial, DDPG_model, steps, semillas_baterias, semillas_mov_obj, generar_animacion = False):
        recompensa_total=0
        df_recompensas = []
        df_estado_acciones = []

        estado = Estado_Determinista(estado_inicial['pos_robots'].copy(), estado_inicial['bateria'].copy(), estado_inicial['pos_obj'].copy(), semillas_mov_obj.copy(), semillas_baterias.copy())

        for j in range(steps):
            estado0 = estado.get_estado_array()
            accion = tf.squeeze(DDPG_model(estado0)).numpy()
            estado.step(np.round(accion), j)
            recompensa_total += estado.evaluar_posiciones()
            df_recompensas.append(estado.evaluar_posiciones())
            df_estado_acciones.append([estado0, accion])
        if(generar_animacion):
            estado.generar_animacion()
        return recompensa_total, df_recompensas, df_estado_acciones

def simulacion_Voraz(estado_inicial, steps, n_robots, semillas_baterias, semillas_mov_obj, generar_animacion = False):
    recompensa_total = 0
    df_recompensas = []
    df_estado_acciones = []

    estado = Estado_Determinista(estado_inicial['pos_robots'].copy(), estado_inicial['bateria'].copy(), estado_inicial['pos_obj'].copy(), semillas_mov_obj.copy(), semillas_baterias.copy())

    for j in range(steps):
        estado0 =  estado.get_estado_experto()
        accion = algoritmo_voraz(estado0, n_robots)
        estado.step(accion, j)
        recompensa_total += estado.evaluar_posiciones()
        df_recompensas.append(estado.evaluar_posiciones())
        df_estado_acciones.append([estado0, accion])
    if(generar_animacion):
        estado.generar_animacion("animacion_voraz.gif")
    return recompensa_total, df_recompensas, df_estado_acciones

def simulacion_GNN_Model(estado_inicial, GNN_model, steps, semillas_baterias, semillas_mov_obj, generar_animacion = False):
    recompensa_total=0
    df_recompensas = []
    df_estado_acciones = []

    estado = Estado_Determinista(estado_inicial['pos_robots'].copy(), estado_inicial['bateria'].copy(), estado_inicial['pos_obj'].copy(), semillas_mov_obj.copy(), semillas_baterias.copy())

    for j in range(steps):
        acciones_robots = np.zeros(len(estado_inicial['pos_robots']))
        estado0 = estado.get_estado()
        atributos, adj_mat, acciones = generar_datos(estado0, acciones_robots)
        adj_list = adjacent_list(adj_mat)
        robots_graphs = create_dataset([[atributos]], [[adj_list]], [[acciones]], len(acciones_robots), for_eval=True)
        for i, graph in enumerate(robots_graphs):
            acciones_robots[i] = np.argmax(tf.squeeze(GNN_model(graph)).numpy())
        estado.step(np.round(acciones_robots),j)
        recompensa_total += estado.evaluar_posiciones()
        df_recompensas.append(estado.evaluar_posiciones())
        df_estado_acciones.append([estado0, acciones_robots])
    if(generar_animacion):
        estado.generar_animacion("animacion_gnn.gif")
    return recompensa_total, df_recompensas, df_estado_acciones


def inicio_evaluacion(SIMULACIONES = 100, DIMENSION = 30, n_robots = 5, n_obj = 6, steps = 50):

    input_graph_spec = pickle.load(open(f'./parametros/data/input_graph_spec.pkl', 'rb'))
    n_robots = int(input("Introduzca el numero de drones: "))
    n_obj = int(input("Introduzca el numero de objetivos a simular: "))
    SIMULACIONES = int(input("Indique el numero de simulaciones: "))
    DIMENSION = int(input("Diga el tamaño de la arena (nxn): "))
    steps = int(input("Pasos de cada simulación: "))

    df_recom_voraz = []
    df_recom_ddpg = []
    df_recom_gnn_voraz = []
    df_recom_gnn_ddpg = []

    df_estado_acciones_voraz = []
    df_estado_acciones_ddpg = []
    df_estado_acciones_gnn_voraz = []
    df_estado_acciones_gnn_ddpg = []

    for i in range(0, SIMULACIONES):
        print("Inicio simulación: ", i)
        semillas_inicio_obj = np.random.randint(1, 29, size=(n_obj,2))
        semillas_inicio_robots = np.random.randint(1, 29, size=(n_robots,2))

        semillas_baterias = np.random.rand(steps)
        semillas_mov_obj = np.random.randint(0,6, size=(steps, n_obj))

        estado_inicial = estado_inicio_det(DIMENSION, n_robots, semillas_inicio_robots.copy(), n_obj, semillas_inicio_obj.copy())

    
        # Algoritmo Voraz
        print("------------ Algoritmo Voraz ----------------")
        recom_voraz = simulacion_Voraz(estado_inicial, steps, n_robots, semillas_baterias, semillas_mov_obj)
        df_recom_voraz.append(recom_voraz[0])
        df_estado_acciones_voraz.append(recom_voraz[2])

        # Modelo DDPG
        print("------------ Modelo DDPG -------------------")
        DDPG_Model = create_actor_model(n_robots, n_obj)
        DDPG_Model.load_weights("./parametros/DDPG/actor_objetivo.h5")
        recom_ddpg = simulacion_DDPG(estado_inicial, DDPG_Model, steps, semillas_baterias, semillas_mov_obj)
        df_recom_ddpg.append(recom_ddpg[0])
        df_estado_acciones_ddpg.append(recom_ddpg[2])

        # Modelo GNN entrenado con Voraz
        print("------------- Modelo GNN con Voraz -----------------")
        GNN_Model = create_GNN_Model(input_graph_spec, 128, 1)
        GNN_Model.load_weights("./parametros/GNN/GNN_Voraz_Model.h5")
        recom_gnn_voraz = simulacion_GNN_Model(estado_inicial, GNN_Model, steps, semillas_baterias, semillas_mov_obj)
        df_recom_gnn_voraz.append(recom_gnn_voraz[0])
        df_estado_acciones_gnn_voraz.append(recom_gnn_voraz[2])

        # Modelo GNN entrenado con DDPG
        print("------------- Modelo GNN con DDPG -----------------")
        GNN_Model = create_GNN_Model(input_graph_spec, 128, 1)
        GNN_Model.load_weights("./parametros/GNN/GNN_DDPG_Model.h5")
        recom_gnn_ddpg = simulacion_GNN_Model(estado_inicial, GNN_Model, steps, semillas_baterias, semillas_mov_obj)
        df_recom_gnn_ddpg.append(recom_gnn_ddpg[0])
        df_estado_acciones_gnn_ddpg.append(recom_gnn_ddpg[2])

        df_recom = [df_recom_voraz, df_recom_gnn_voraz, df_recom_ddpg, df_recom_gnn_ddpg]

def evaluacion_libre(DIMENSION = 30, n_robots = 5, n_obj = 6, steps = 50, version_modelo = 'voraz'):
    
    input_graph_spec = pickle.load(open(f'./parametros/data/input_graph_spec.pkl', 'rb'))
    n_robots = int(input("Introduzca el numero de drones: "))
    n_obj = int(input("Introduzca el numero de objetivos a simular: "))
    DIMENSION = int(input("Diga el tamaño de la arena (nxn): "))
    steps = int(input("Pasos de cada simulación: "))

    df_recom_voraz = []
    df_recom_ddpg = []
    df_recom_gnn_voraz = []
    df_recom_gnn_ddpg = []

    df_estado_acciones_voraz = []
    df_estado_acciones_ddpg = []
    df_estado_acciones_gnn_voraz = []
    df_estado_acciones_gnn_ddpg = []

    if(version_modelo == 'voraz'):
        semillas_inicio_obj = np.random.randint(1, 29, size=(n_obj,2))
        semillas_inicio_robots = np.random.randint(1, 29, size=(n_robots,2))

        semillas_baterias = np.random.rand(steps)
        semillas_mov_obj = np.random.randint(0,6, size=(steps, n_obj))

        estado_inicial = estado_inicio_det(DIMENSION, n_robots, semillas_inicio_robots.copy(), n_obj, semillas_inicio_obj.copy())

        # Algoritmo Voraz
        print("------------ Algoritmo Voraz ----------------")
        recom_voraz = simulacion_Voraz(estado_inicial, steps, n_robots, semillas_baterias, semillas_mov_obj, generar_animacion=True)
        df_recom_voraz.append(recom_voraz[0])
        df_estado_acciones_voraz.append(recom_voraz[2])

        # Modelo GNN entrenado con Voraz
        print("------------- Modelo GNN con Voraz -----------------")
        GNN_Model = create_GNN_Model(input_graph_spec, 128, 1)
        GNN_Model.load_weights("./parametros/GNN/GNN_Voraz_Model.h5")
        recom_gnn_voraz = simulacion_GNN_Model(estado_inicial, GNN_Model, steps, semillas_baterias, semillas_mov_obj, generar_animacion=True)
        df_recom_gnn_voraz.append(recom_gnn_voraz[0])
        df_estado_acciones_gnn_voraz.append(recom_gnn_voraz[2])

        df_recom = [df_recom_voraz, df_recom_gnn_voraz]

    elif(version_modelo=='ddpg'):
        # Modelo DDPG
        print("------------ Modelo DDPG -------------------")
        DDPG_Model = create_actor_model(n_robots, n_obj)
        DDPG_Model.load_weights("./parametros/DDPG/actor_objetivo.h5")
        recom_ddpg = simulacion_DDPG(estado_inicial, DDPG_Model, steps, semillas_baterias, semillas_mov_obj, generar_animacion=True)
        df_recom_ddpg.append(recom_ddpg[0])
        df_estado_acciones_ddpg.append(recom_ddpg[2])

        # Modelo GNN entrenado con DDPG
        print("------------- Modelo GNN con DDPG -----------------")
        GNN_Model = create_GNN_Model(input_graph_spec, 128, 1)
        GNN_Model.load_weights("./parametros/GNN/GNN_DDPG_Model.h5")
        recom_gnn_ddpg = simulacion_GNN_Model(estado_inicial, GNN_Model, steps, semillas_baterias, semillas_mov_obj, generar_animacion=True)
        df_recom_gnn_ddpg.append(recom_gnn_ddpg[0])
        df_estado_acciones_gnn_ddpg.append(recom_gnn_ddpg[2])

        df_recom = [df_recom_voraz, df_recom_gnn_voraz]
    else:
        print("ingrese una versión de modelo valida")