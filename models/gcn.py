import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def sparseTensorFromMatrix(sparse_mx):
    """Convert sparse matrix to tuple representation"""
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    res = tf.sparse.SparseTensor(coords, values, shape)
    return tf.cast(res, dtype=tf.float32)


def sparseDropout(x, dropout_rate, noise_shape):
    retain_rate = 1 - dropout_rate
    random_tensor = retain_rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor),
                           dtype=tf.bool)
    unnormd = tf.sparse.retain(x, dropout_mask)
    return unnormd * (1. / retain_rate)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) initialization"""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(shape, minval=-init_range,
                                maxval=init_range)
    return tf.Variable(initial, name=name, dtype=tf.float32)


def zeros(shape, name=None):
    initial = tf.zeros(shape)
    return tf.Variable(initial, name=name, dtype=tf.float32)


class GCN(tf.keras.Model):
    def __init__(self, in_dim, out_dim,
                 nonzero_feat_shape,
                 convolution_matrix,
                 hidden_layer_dim=16,
                 dropout_rate=0.5):
        super(GCN, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonzero_feat_shape = nonzero_feat_shape
        self.convolution_matrix = sparseTensorFromMatrix(convolution_matrix)
        self.hidden_layer_dim = hidden_layer_dim
        self.dropout_rate = dropout_rate
        self.weight_matrix1 = glorot(shape=[in_dim,
                                            hidden_layer_dim],
                                     name="weights1")
        self.weight_matrix2 = glorot(shape=[hidden_layer_dim,
                                            out_dim],
                                     name="weights2")
        self.bias_vector1 = zeros([hidden_layer_dim],
                                  name="bias1")
        self.bias_vector2 = zeros([out_dim], name="bias2")

    def call(self, x):
        # Layer 1: inputs are sparse
        x = sparseTensorFromMatrix(x)
        x = sparseDropout(x, self.dropout_rate,
                          self.nonzero_feat_shape)
        x = tf.sparse.sparse_dense_matmul(x, self.weight_matrix1)
        x = tf.sparse.sparse_dense_matmul(self.convolution_matrix, x)
        x = x + self.bias_vector1
        x = tf.nn.relu(x)
        # Layer 2: inputs are not sparse anymore and
        # for some reason we do no relu
        x = tf.nn.dropout(x, 1 - self.dropout_rate)
        x = tf.matmul(x, self.weight_matrix2)
        x = tf.sparse.sparse_dense_matmul(self.convolution_matrix, x)
        x = x + self.bias_vector2
        return x
