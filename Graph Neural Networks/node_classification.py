import utils
import data_utils
import numpy as np
import networkx as nx
import customlayer as cl
import tensorflow as tf

DATA_SET = ["Cora", "Citeseer"]
# Hyper-parameters
learning_rate = 0.01
epochs = 100
num_units = 32
dropout_rate = 0.6


def node_classification():
    for data in DATA_SET:
        train = nx.read_gpickle("./datasets/"+data+"_Train/data.pkl")[0]
        test = nx.read_gpickle("./datasets/"+data+"_Eval/data.pkl")[0]

        x = data_utils.get_node_attributes(train)
        y = data_utils.get_node_labels(train) - 1

        num_of_labels = len(set(y))

        norm_adj_matrix = utils.get_normalized_adjacency_matrix(train).astype(np.float32)
        norm_adj_matrix_eval = utils.get_normalized_adjacency_matrix(test).astype(np.float32)

        cl.GraphConvLayer.normalized_adj = norm_adj_matrix

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(np.shape(x)))
        model.add(cl.GraphConvLayer(num_units, activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        model.add(cl.GraphConvLayer(num_units, activation="relu"))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
        model.add(cl.GraphConvLayer(activation="softmax", units=num_of_labels))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor="accuracy", patience=50, restore_best_weights=True)

        model.fit(np.expand_dims(x, axis=0), np.expand_dims(y, axis=0), epochs=epochs, callbacks=[early_stopping])

        cl.GraphConvLayer.normalized_adj = norm_adj_matrix_eval
        x_eval = data_utils.get_node_attributes(test)
        y_eval = data_utils.get_node_labels(test) - 1

        score = model.evaluate(tf.expand_dims(x_eval, axis=0), tf.expand_dims(y_eval, axis=0), verbose=0)
        print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')


if __name__ == '__main__':
    node_classification()
