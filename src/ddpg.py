'''Definición del sistema experto para generación de los datos de entrenamiento del sistema de inteligencia distribuida.
En él, se definirá el MDP el algoritmo de Deep-RL que lo resolverá (DDPG) y finalmente una fución para generar (en un txt, csv....)
el conjutno de datos de entramiento que serán simplemente el conjunto de estados explorados total y la matriz de acciones elegidas para el conjunto
según el estado. Este será preprocesado y transformado por el gnn_data_setup
'''
import math
import numpy as np
from scipy import stats
import random
import tensorflow as tf
from simulador_entorno import Estado, estado_inicio, carga_bateria, descarga_bateria
from gnn_data_setup import *

#Definimos las arquitecturas de las dos redes Actor-Critic

def create_actor_model(n_robots, n_obj):
    ini_pesos = tf.random_uniform_initializer(minval=-0.005, maxval=0.005)
    input = tf.keras.layers.Input(shape=(4*n_robots,))  #el segundo n_robots es por las baterías; los 2* es porque cada robot tiene 2 valores en coordenadas
    d1 = tf.keras.layers.Dense(252, activation="relu")(input)
    d2 = tf.keras.layers.Dense(128, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(64, activation="relu")(d2)
    output_layer = tf.keras.layers.Dense(n_robots, activation="sigmoid", kernel_initializer=ini_pesos)(d3)
    #output_layer = tf.keras.layers.Dense(n_robots)(d4)

    output = output_layer*6
    #output = tf.keras.activations.relu(output_layer, max_value=6)

    return  tf.keras.Model(input, output)


def create_critic_model(n_robots, n_obj):
    # entrada del estado
    estado_input = tf.keras.Input(shape=(4*n_robots))
    estado_output = tf.keras.layers.Dense(64, activation = "relu")(estado_input)
    estado_output = tf.keras.layers.Dense(128, activation="relu")(estado_output)

    # entrada de las acciones
    acciones_input = tf.keras.Input(shape=(n_robots))
    acciones_output = tf.keras.layers.Dense(64, activation="relu")(acciones_input)
    acciones_output = tf.keras.layers.Dense(128, activation="relu")(acciones_output)

    inputs = tf.keras.layers.Concatenate()([estado_output, acciones_output])

    #resto de la acquitectura de la red
    d1 = tf.keras.layers.Dense(252, activation="relu")(inputs)
    d2 = tf.keras.layers.Dense(128, activation="relu")(d1)
    d3 = tf.keras.layers.Dense(64, activation="relu")(d2)
    output = tf.keras.layers.Dense(1)(d3)

    return tf.keras.Model([estado_input, acciones_input], output)

class TrainBuffer:
    def __init__(self, capacidad, batch_size, n_robots, GAMMA):
        self.capacidad = capacidad
        self.batch = batch_size
        self.GAMMA = GAMMA
        self.contador = 0

        self.buffer_estados = np.zeros((self.capacidad, 4*n_robots))
        self.buffer_acciones = np.zeros((self.capacidad, n_robots))
        self.buffer_recompensa = np.zeros((self.capacidad,1))
        self.buffer_sig_estado = np.zeros((self.capacidad, 4*n_robots))

    def record(self, obs):
        # cuando se sobrepasa la capacidad, se empiezan a sustituir los antiguos registros
        index = self.contador % self.capacidad

        self.buffer_estados[index] = obs[0]
        self.buffer_acciones[index] = obs[1]
        self.buffer_recompensa[index] = obs[2]
        self.buffer_sig_estado[index] = obs[3]

        self.contador += 1

    @tf.function
    def update(self, batch_estado, batch_acciones, batch_recompensa, batch_sig_estado, actor, critico, actor_objetivo, critic_objetivo, optimizador_critico, optimizador_actor):

        #entrenamiento de la red
        with tf.GradientTape() as tape:
            acciones_obj = actor_objetivo(batch_sig_estado, training=True)
            y = batch_recompensa + self.GAMMA*critic_objetivo([batch_sig_estado, acciones_obj], training=True)
            res_critico = critico([batch_estado, batch_acciones], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - res_critico))

        grad_critic = tape.gradient(critic_loss, critico.trainable_variables)
        optimizador_critico.apply_gradients(zip(grad_critic, critico.trainable_variables))

        with tf.GradientTape() as tape:
            acciones = actor(batch_estado, training=True)
            res_critico = critico([batch_estado, acciones], training=True)

            actor_loss = -tf.math.reduce_mean(res_critico)

        grad_actor = tape.gradient(actor_loss, actor.trainable_variables)
        optimizador_actor.apply_gradients(zip(grad_actor, actor.trainable_variables))

    def learn(self, actor, critico, actor_objetivo, critic_objetivo, optimizador_critico, optimizador_actor):

        record_range = min(self.contador, self.capacidad)
        # se genera un batch aleatorio para el aprendizaje
        batch_indices = np.random.choice(record_range, self.batch)

        batch_estados = tf.convert_to_tensor(self.buffer_estados[batch_indices])
        batch_acciones = tf.convert_to_tensor(self.buffer_acciones[batch_indices])
        batch_recompensa = tf.convert_to_tensor(self.buffer_recompensa[batch_indices])
        batch_recompensa = tf.cast(batch_recompensa, dtype=tf.float32)
        batch_sig_estado = tf.convert_to_tensor(self.buffer_sig_estado[batch_indices])
        self.update(batch_estados, batch_acciones, batch_recompensa, batch_sig_estado, actor, critico, actor_objetivo, critic_objetivo, optimizador_critico, optimizador_actor)


#según el pseudocodigo
@tf.function
def update_target_weights(pesos_obj, pesos, TAU):
    for (a, b) in zip(pesos_obj, pesos):
        a.assign(b * TAU + a * (1 - TAU))

def policy3(estado, step):
  acciones = tf.squeeze(actor(estado))
  if(random.random() < math.pow(2, -step/40)):
    return np.random.rand(n_robots)
  num = np.random.randint(0, n_robots-1, size=2)
  rand_a = np.random.rand(2)
  acciones_proc = acciones.numpy()
  acciones_proc[num[0]], acciones_proc[num[1]] = rand_a[0], rand_a[1]

  return acciones_proc

def policy2(estado, step):
  acciones = tf.squeeze(actor(estado))
  # se agrega ruido mediante el adjacent swap de dos ordenes
  if(random.random() < math.pow(2, -step/40)):
    return np.random.rand(n_robots)
  num = random.randint(0,n_robots-2)
  acciones_proc = acciones.numpy()
  acciones_proc[num], acciones_proc[num+1] = acciones_proc[num+1], acciones_proc[num]

  return acciones_proc

def policy1(estado, step, actor, n_robots):
  #acciones = tf.squeeze(actor(normalizar_vec(estado)))
  acciones = tf.squeeze(actor(estado))
  if(random.random() < math.pow(2, -step/50)):
    return np.random.rand(n_robots)

  return acciones

def normalizar_vec(vec):
  vec_min = tf.math.reduce_min(vec)
  vec_max = tf.math.reduce_max(vec)
  return (vec-vec_min)/(vec_max-vec_min)



def entrenar_ddpg(GAMMA = 0.8, SIMULACIONES=60000, DIMMENSION=30, TAU=0.005, n_robots=5, n_obj=6, preentrenda=False):
  estado_inicial = estado_inicio(DIMMENSION, n_robots, n_obj)
  estado = Estado(estado_inicial['pos_robots'].copy(), estado_inicial['bateria'].copy(), estado_inicial['pos_obj'].copy())

  if(preentrenda):
      actor = create_actor_model(n_robots, n_obj)
      critico = create_critic_model(n_robots, n_obj)
      actor_objetivo = create_actor_model(n_robots, n_obj)
      critic_objetivo = create_critic_model(n_robots, n_obj)

      actor.load_weights("./parametros/DDPG/preentrenada/actor.h5")
      critico.load_weights("./parametros/DDPG/preentrenada/critico.h5")
      actor_objetivo.load_weights("./parametros/DDPG/preentrenada/actor_objetivo.h5")
      critic_objetivo.load_weights("./parametros/DDPG/preentrenada/critico_objetivo.h5")
  else:
    #Declaramos los modelos con la misma arquitectura
      actor = create_actor_model(n_robots, n_obj)
      critico = create_critic_model(n_robots, n_obj)

      actor_objetivo = create_actor_model(n_robots, n_obj)
      critic_objetivo = create_critic_model(n_robots, n_obj)

      # le ponemos los mismos pesos
      actor_objetivo.set_weights(actor.get_weights())
      critic_objetivo.set_weights(critico.get_weights())

      optimizador_actor = tf.keras.optimizers.Adam(0.0001)
      optimizador_critico = tf.keras.optimizers.Adam(0.0002)



  buffer = TrainBuffer(100000, 128, n_robots, GAMMA)
  recompensas = []
  simulaciones = []
  #Train Loop

  for j in range(SIMULACIONES):
    estado_inicial = estado_inicio(DIMMENSION, n_robots, n_obj)
    estado = Estado(estado_inicial['pos_robots'].copy(), estado_inicial['bateria'].copy(), estado_inicial['pos_obj'].copy())
    print("Estado inicial: ", estado.get_estado_array())
    estado0 = estado.get_estado_array()
    recompensa_por_simulacion = 0
    acciones = []
    for i in range(50):

        accion = policy1(estado0, i, actor, n_robots)
        accion_proc = np.round(accion)
        estado.step(accion_proc)
        estado1 = estado.get_estado_array()
        recompensa = estado.evaluar_posiciones()
        accion = np.array([accion])
        buffer.record((estado0, accion, recompensa, estado1))
        buffer.learn(actor, critico, actor_objetivo, critic_objetivo, optimizador_critico, optimizador_actor)
        estado0 = estado1.copy()
        acciones.append(accion_proc)

        update_target_weights(actor_objetivo.variables, actor.variables, TAU)
        update_target_weights(critic_objetivo.variables, critico.variables, TAU)
        recompensa_por_simulacion += recompensa
    print("Estado final: ", estado1)
    print("Mean acciones:", np.mean(acciones, axis=0))
    print("Mode acciones:", stats.mode(acciones)[0], " count array", stats.mode(acciones)[1])
    #print("Simulacion", j)
    #print("Pesos actor: ", actor.variables)
    #print("Pesos actor objetivo: ", actor_objetivo.variables)

    recompensas.append(recompensa_por_simulacion)
    simulaciones.append(j)
    print("Simulación ", j, " recompensa: ", recompensa_por_simulacion)


  #Guardar los pesos
  actor.save_weights("actor.h5")
  actor_objetivo.save_weights("actor_objetivo.h5")

  critico.save_weights("critico.h5")
  critic_objetivo.save_weights("critico_objetivo.h5")