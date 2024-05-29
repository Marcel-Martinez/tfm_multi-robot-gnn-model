import tensorflow as tf
import tensorflow_gnn as tfgnn
import pickle
import numpy as np

def create_graph_spec(n_robots):
  return tfgnn.GraphTensorSpec.from_piece_specs(
      context_spec=tfgnn.ContextSpec.from_field_specs(features_spec={
                    'context': tf.TensorSpec(shape=(1,7), dtype=tf.int32)
      }),
      node_sets_spec={
          'dron':
              tfgnn.NodeSetSpec.from_field_specs(
                  features_spec={
                      'vector_atributos':
                          #tf.TensorSpec((None, 13), tf.int32)
                          tf.TensorSpec((None, 7), tf.int32)
                  },
                  sizes_spec=tf.TensorSpec((1,), tf.int32))
      },
      edge_sets_spec={
          'conexion':
              tfgnn.EdgeSetSpec.from_field_specs(
                  sizes_spec=tf.TensorSpec((1,), tf.int32),
                  adjacency_spec=tfgnn.AdjacencySpec.from_incident_node_sets(
                      'dron', 'dron'))
      })

def load_pickles(path='./data/data_', num_data_files=1):
  atributos_data = []
  adj_mat_data = []
  acciones_data = []
  for i in range(1,num_data_files+1):
    fil = open(path+str(i)+'.pkl', 'rb')
    data = pickle.load(fil)
    adj_mat_data.append(data[0])
    atributos_data.append(data[1])
    acciones_data.append(data[2])

  return atributos_data, adj_mat_data, acciones_data

def adjacent_list(adj_mat):
  adj_mat = adj_mat>0
  adj_mat = adj_mat.astype(int)
  adj_list = []
  for i in range(len(adj_mat)):
    for j in range(len(adj_mat)):
      if(i!=j and adj_mat[i,j]==1):
          adj_list.append([i,j])
  return np.array(adj_list)

def create_GraphTensor(atributos, adj_list, accion, for_eval=False):
  vecinos = len(atributos[:,0])
  if(for_eval):
    Accion = tf.keras.utils.to_categorical([accion], num_classes=7)
  else:
    Accion = tf.keras.utils.to_categorical(accion, num_classes=7)

  dron = tfgnn.NodeSet.from_fields(sizes=[vecinos],
      features={'vector_atributos': atributos.astype('int32')})

  adj_drones = tfgnn.Adjacency.from_indices(source=('dron', np.zeros(len(adj_list)).astype('int32')),
                                                target=('dron', np.arange(1, len(adj_list)+1).astype('int32')))

  conexion = tfgnn.EdgeSet.from_fields(sizes=[len(adj_list)], adjacency=adj_drones)

  context = tfgnn.Context.from_fields(features={'context': Accion.astype('int32')})

  return tfgnn.GraphTensor.from_pieces(node_sets={'dron': dron}, edge_sets={'conexion': conexion}, context=context)


def create_dataset(atributos, adj_list, acciones, n_robots, for_eval=False):
  ds = []
  atr = []
  mat = []
  acc = []
  lista_robots = np.arange(0, n_robots)
  for sim in range(0, len(atributos)):
    for step in range(0, len(atributos[sim])):
      if(len(adj_list[sim][step]) == 0):
        for robot in range(0, len(atributos[sim][step])):
          atr.append(np.array([atributos[sim][step][robot]]))
          mat.append(np.array([]))
          acc.append(acciones[sim][step][robot])
      else:
        robots_no_conectados = np.delete(lista_robots, np.unique(adj_list[sim][step][:,0]))
        for robot in robots_no_conectados:
          atr.append(np.array([atributos[sim][step][robot]]))
          mat.append(np.array([]))
          acc.append(acciones[sim][step][robot])
        for robot in np.unique(adj_list[sim][step][:,0]):
          mat.append(adj_list[sim][step][np.where(adj_list[sim][step][:,0] == robot)[0]])
          atr.append(atributos[sim][step][np.unique(adj_list[sim][step])])
          acc.append(acciones[sim][step][robot])

  for i in range(0, len(mat)):
    ds.append(create_GraphTensor(atr[i], mat[i], acc[i], for_eval))

  return ds

def tfrecord_decode_fn(record):
  graph_spec = pickle.load(open(f'./parametros/data/graph_spec.pkl', 'rb'))
  graph_record = tfgnn.parse_single_example(graph_spec, record, validate=True)
  feat = graph_record.context.get_features_dict()
  accion = feat.pop('context')[0]
  output_graph = graph_record.replace_features(context=feat)
  return output_graph, accion


# Modelo GNN

def create_GNN_Model(graph_spec, message_dim, num_message_passing):
    input_graph = tf.keras.layers.Input(type_spec=graph_spec)
    graph = input_graph.merge_batch_to_components()

    def set_initial_node_state(node_set, node_set_name):
      return {'vector_atributos': tf.keras.Sequential([
              tf.keras.layers.Dense(32, activation='relu'),
              tf.keras.layers.Dense(64, activation='relu'),
              tf.keras.layers.Dense(128, activation='relu'),
              tf.keras.layers.Dense(256, activation='relu'),
              tf.keras.layers.Dense(128, activation='relu')])(node_set['vector_atributos'])}

    graph = tfgnn.keras.layers.MapFeatures(node_sets_fn=set_initial_node_state)(graph)

    def dense_SOURCE():
      return tf.keras.Sequential([tf.keras.layers.Dense(message_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
                                  tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))])
    def dense_TARGET():
      return tf.keras.Sequential([tf.keras.layers.Dense(message_dim, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.0005)),
                                  tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))])

    for i in range(num_message_passing):
      graph = tfgnn.keras.layers.GraphUpdate(
          node_sets={"dron": tfgnn.keras.layers.NodeSetUpdate(
              {
                  "conexion": tfgnn.keras.layers.SimpleConv(
                      sender_node_feature="vector_atributos",
                      message_fn = dense_TARGET(),
                      reduce_type = "sum",
                      receiver_tag=tfgnn.TARGET,
                      receiver_feature="vector_atributos"
                  )
              }
          ,tfgnn.keras.layers.NextStateFromConcat(dense_TARGET()), node_input_feature="vector_atributos")}
      )(graph)
      graph = tfgnn.keras.layers.GraphUpdate(
          node_sets={"dron": tfgnn.keras.layers.NodeSetUpdate(
              {#se elige el tipo de arco sobre el que se hará la conv, en este caso sobre el unico tipo que es conexion dron-dron
                  "conexion": tfgnn.keras.layers.SimpleConv(
                      sender_node_feature="vector_atributos",
                      message_fn = dense_SOURCE(),
                      reduce_type = "sum",
                      receiver_tag=tfgnn.SOURCE,
                      receiver_feature="vector_atributos"
                  )
              }
          ,tfgnn.keras.layers.NextStateFromConcat(dense_SOURCE()), node_input_feature="vector_atributos")}
      )(graph)

    # volvemos a gener el vector de atributos de los arcos a partir del traspaso de información de los nodos transformados en el SimpleConv a sus arcos vecinos
    # Necesario para el paso del Pool

    pooled_features = tfgnn.keras.layers.Pool(tfgnn.CONTEXT, "sum", node_set_name='dron', feature_name='vector_atributos')(graph)
    d1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(pooled_features)
    d2 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))(d1)
    output = tf.keras.layers.Dense(7, activation = "softmax")(d2)

    return tf.keras.Model([input_graph], output)