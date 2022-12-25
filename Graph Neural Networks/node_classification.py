import data_utils as utils
import numpy as np
import networkx as nx
import customlayer as cl
from tensorflow.python.keras import activations
import tensorflow as tf


def get_normalized_adjacency_matrix(graph):
    adjacency_matrix = utils.get_adjacency_matrix(graph)
    d_sqrt = np.sqrt(np.diag(np.sum(adjacency_matrix, axis=1)))
    d_sqrt_inverse = np.linalg.inv(d_sqrt) if np.linalg.det(d_sqrt) else np.linalg.pinv(d_sqrt)
    return np.matmul(np.matmul(d_sqrt_inverse, adjacency_matrix), d_sqrt_inverse)


cora = nx.read_gpickle("./datasets/Cora_train/data.pkl")[0]
cora_eval = nx.read_gpickle("./datasets/Cora_Eval/data.pkl")[0]

X = utils.get_node_attributes(cora)
Y = utils.get_node_labels(cora)-1

norm_adj_matrix = get_normalized_adjacency_matrix(cora).astype(np.float32)
norm_adj_matrix_eval = get_normalized_adjacency_matrix(cora_eval).astype(np.float32)

cl.GraphConvLayer.normalized_adj = norm_adj_matrix

model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(np.shape(X)))
model.add(cl.GraphConvLayer(32, activation="relu"))
model.add(tf.keras.layers.Dropout(rate=0.7))
model.add(cl.GraphConvLayer(32, activation="relu"))
model.add(tf.keras.layers.Dropout(rate=0.7))
model.add(cl.GraphConvLayer(activation=activations.softmax, units=7))
model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=50, restore_best_weights=True)

model.fit(np.expand_dims(X, axis=0), np.expand_dims(Y, axis=0), epochs=200, callbacks=[early_stopping])


cl.GraphConvLayer.normalized_adj = norm_adj_matrix_eval
X_eval = utils.get_node_attributes(cora_eval)
Y_eval = utils.get_node_labels(cora_eval)-1

score = model.evaluate(tf.expand_dims(X_eval, axis=0), tf.expand_dims(Y_eval, axis=0), verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
