import pickle
from simulador_entorno import Estado, estado_inicio, carga_bateria, descarga_bateria
from gnn_data_setup import *
from ddpg import *
from greedys_algorithms import algoritmo_voraz
from gnn_model import *
import tensorflow as tf
import tensorflow_gnn as tfgnn


def simulacion_DDPG(estado_inicial, DDPG_model, steps):
    recompensa_total=0
    df_atributos = []
    df_adj_mat = []
    df_acciones = []

    estado = Estado(estado_inicial['pos_robots'].copy(), estado_inicial['bateria'].copy(), estado_inicial['pos_obj'].copy())

    for j in range(steps):
        estado0 = estado.get_estado_array()
        datos_estado0 = estado.get_estado()
        accion = tf.squeeze(DDPG_model(estado0)).numpy()
        estado.step(np.round(accion))
        atributos, adj_mat, acciones = generar_datos(datos_estado0, np.round(accion))
        df_atributos.append(atributos)
        df_adj_mat.append(adj_mat)
        df_acciones.append(acciones)
        recompensa_total += estado.evaluar_posiciones()

    print(recompensa_total)
    return [np.array(df_adj_mat), df_atributos, np.array(df_acciones)]



def simulacion_Experta(estado_inicial, steps, n_robots):
    recompensa_total=0
    df_atributos = []
    df_adj_mat = []
    df_acciones = []

    estado = Estado(estado_inicial['pos_robots'].copy(), estado_inicial['bateria'].copy(), estado_inicial['pos_obj'].copy())

    for j in range(steps):
        pos, baterias, LiDAR, objetivos_cubiertos = estado.get_estado_experto()
        datos_estado0 = estado.get_estado()
        accion = algoritmo_voraz([pos.copy(), baterias.copy(), LiDAR.copy(), objetivos_cubiertos.copy()], n_robots)
        estado.step(accion)
        atributos, adj_mat, acciones = generar_datos(datos_estado0, accion)
        df_atributos.append(atributos)
        df_adj_mat.append(adj_mat)
        df_acciones.append(acciones)
        recompensa_total += estado.evaluar_posiciones()

    print(recompensa_total)
    return [np.array(df_adj_mat), df_atributos, np.array(df_acciones)]


def entrenamiento_ddpg():
    # entrenar modelo DDPG
    res = input("Desea utilizar una DDPG preentrenada y continuar su entrenamiento? (Si/No):")
    if(res=='Si'):
        print("Indicar los parametros del entrenamiento del modelo DDPG: ")
        GAMMA = float(input("GAMMA="))
        SIMULACIONES = int(input("SIMULACIONES="))
        DIMMENSION = int(input("DIMENSION="))
        TAU = float(input("TAU="))
        n_robots = int(input("Numero de robots="))
        n_obj = int(input("Numero de objetivos="))
        entrenar_ddpg(GAMMA = GAMMA, SIMULACIONES=SIMULACIONES, DIMMENSION=DIMMENSION, TAU=TAU, n_robots=n_robots, n_obj=n_obj, preentrenda=True)
    else:
        print("Indicar los parametros del entrenamiento del modelo DDPG: ")
        GAMMA = float(input("GAMMA="))
        SIMULACIONES = int(input("SIMULACIONES="))
        DIMMENSION = int(input("DIMENSION="))
        TAU = float(input("TAU="))
        n_robots = int(input("Numero de robots="))
        n_obj = int(input("Numero de objetivos="))
        entrenar_ddpg(GAMMA = GAMMA, SIMULACIONES=SIMULACIONES, DIMMENSION=DIMMENSION, TAU=TAU, n_robots=n_robots, n_obj=n_obj)

def generar_datos_GNN(n_robots=5, n_obj=6, SIMULACIONES=100, DIMMENSION=30, steps=50):
    #generar datos de entrenamiento para la GNN comenzando por la DDPG
    modelo = input("Con cual modelo desea entrenar la GNN? ('ddpg entrenada', 'voraz') ")
    n_robots = int(input("Introduzca el numero de drones: "))
    n_obj = int(input("Introduzca el numero de objetivos a simular: "))
    SIMULACIONES = int(input("Indique el numero de simulaciones: "))
    DIMMENSION = int(input("Diga el tamaño de la arena (nxn): "))
    steps = int(input("Pasos de cada simulación: "))
    if(modelo == 'ddpg'):
        DDPG_model = create_actor_model(n_robots, n_obj)
        # cargamos los pesos del modelo
        try:
            DDPG_model.load_weights("./parametros/DDPG/actor_objetivo.h5")
        except:
            print("La carpeta de parametros/DDPG/ no existe o no contiene un modelo en el formato correcto")
        finally:
            for i in range(SIMULACIONES):
                estado_inicial = estado_inicio(DIMMENSION, n_robots, n_obj)
                pickle.dump(simulacion_DDPG(estado_inicial, DDPG_model, steps), open(f'./data/DDPG/data_{i+1}.pkl', 'wb'))
    else:
    # se sigue con la solucion voraz
        for i in range(SIMULACIONES):
            estado_inicial = estado_inicio(DIMMENSION, n_robots, n_obj)
            pickle.dump(simulacion_Experta(estado_inicial, steps, n_robots), open(f'./data/Voraz/data_{i+1}.pkl', 'wb'))

    graph_spec = create_graph_spec(n_robots)

    #cargamos los pickles
    atributos, adj_mat, acciones = load_pickles(path='data/Voraz/data_', num_data_files=SIMULACIONES)
    adj_list = [[] for i in range(len(adj_mat))]

    # convertimos todas las adj_mat a listas de adjacencia (adj_list)
    for simulacion, mat in enumerate(adj_mat):
        for paso_simulacion in range(0, len(adj_mat[simulacion])):
            adj_list[simulacion].append(adjacent_list(mat[paso_simulacion]))
    print("Creando dataset de GraphTensors....")
    dataset = create_dataset(atributos, adj_list, acciones, n_robots)

    # realizar la grabación de TFRecords con lo que retornará esta función (la lista dataset generada arriba)
    #filename = f'train_data/train_data_ddpg.tfrecord'
    filename = f'train_data/train_data.tfrecord'
    with tf.io.TFRecordWriter(filename) as writer:
        for graph_tensor in dataset:
            tf_example = tfgnn.write_example(graph_tensor)
            writer.write(tf_example.SerializeToString())
    print("Dataset de GraphTensors creado y guardado en train_data/train_data.tfrecord")


def entrenamiento_gnn(n_robots = 5):
    print("Iniciando entrenamiento de la GNN (leyendo datos)...")
    graph_spec = create_graph_spec(n_robots)
    pickle.dump(graph_spec, open(f'./parametros/data/graph_spec.pkl', 'wb'))
    tf_dataset = tf.data.TFRecordDataset(['train_data/train_data.tfrecord']).map(tfrecord_decode_fn)

    #guardamos los graph_spec como parametros, porque nos interesan ambos
    print("Guardando backup del graph_spec generado")
    input_graph_spec, accion_spec = tf_dataset.element_spec
    pickle.dump(input_graph_spec, open(f'./parametros/data/input_graph_spec.pkl', 'wb'))

    GNN_Model = create_GNN_Model(input_graph_spec, 128, 1) 
    tf_dataset = tf_dataset.shuffle(64)
    size = len(list(tf_dataset))
    train_size = int(0.6 * size)
    val_size = int(0.2 * size)
    test_size = int(0.2 * size)

    # Dividir los datos
    train_dataset = tf_dataset.take(train_size)
    test_dataset = tf_dataset.skip(train_size)
    val_dataset = test_dataset.skip(val_size)
    test_dataset = test_dataset.take(test_size)
    print("Creando batchs de entrenamiento")
    #crear batchs
    train_dataset = train_dataset.batch(64)
    test_dataset = test_dataset.batch(64)
    val_dataset = val_dataset.batch(64)
    print("Iniciando entrenamiento")
    #Entrenamiento con Cosine learning decay
    epochs = 1000
    total_steps = 235
    initial_learning_rate = 0.002
    final_learning_rate = 0.0001
    lr_scheduler = tf.keras.optimizers.schedules.CosineDecay(initial_learning_rate, epochs*total_steps, warmup_target=final_learning_rate)
    loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    optimizador = tf.keras.optimizers.Adam(learning_rate=lr_scheduler)
    GNN_Model.compile(optimizador, loss=loss_func, metrics=['accuracy'])
    GNN_Model.fit(train_dataset, epochs=1000, validation_data=val_dataset)

    #Guardamos los pesos
    GNN_Model.save_weights("./parametros/GNN/GNN_DDPG_Model.h5")
    print("Finalizado... Guardado el modelo en parametros/GNN/")