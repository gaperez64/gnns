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
                                maxval=init_range,
                                dtype=tf.float32)
    return tf.Variable(initial, name=name, dtype=tf.float32,
                       trainable=True)
