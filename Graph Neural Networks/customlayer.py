from keras.layers import Layer
import tensorflow as tf
from tensorflow.python.keras import layers
import numpy as np
import data_utils as utils


class GraphConvLayer(Layer):
    normalized_adj = None

    def __init__(self, units: int, activation: str = None, **kwargs: object) -> object:
        super(GraphConvLayer, self).__init__()
        self.units = units
        self.w = None
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        w_init = tf.keras.initializers.glorot_normal()
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units)), dtype="float32", trainable=True)

    def call(self, inputs, training=None):
        inputs = tf.cast(inputs, dtype="float32")
        return self.activation(tf.matmul(tf.matmul(GraphConvLayer.normalized_adj, inputs, output_type="float32"), self.w, output_type="float32"), )
