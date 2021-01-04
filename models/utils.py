import networkx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf


def adjTensor(graph):
    adj = networkx.adjacency_matrix(graph)
    return tf.Tensor(adj, dtype=tf.float32)


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


def zeros(shape, name=None):
    """All zeros"""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name, dtype=tf.float32,
                       trainable=True)


def normalizedAdj(adj, adj_replacement=None):
    """Symmetrically normalize adjacency matrix"""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # We have D, we can now compute DAD
    if adj_replacement is None:
        adj_replacement = adj
    return adj_replacement.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def normdAdj(graph):
    """
    Recover the symmetrically-normalized adjacency matrix
    """
    adj = networkx.adjacency_matrix(graph)
    return normalizedAdj(adj)


def normdAdjId(graph, scaling_factor=None):
    """
    Recover the symmetrically-normalized adjacency matrix
    with an identity added
    """
    adj = networkx.adjacency_matrix(graph)
    # adding the identity matrix
    adj = adj + sp.eye(adj.shape[0])
    adj_id_scaled = None
    if scaling_factor is not None:
        adj_id_scaled = adj + (scaling_factor * sp.eye(adj.shape[0]))
    return normalizedAdj(adj, adj_replacement=adj_id_scaled)


def adjIdTensor(graph, scaling_factor=None):
    adj = networkx.adjacency_matrix(graph)
    if scaling_factor is None:
        adj = adj + sp.eye(adj.shape[0])
    else:
        adj = adj + (scaling_factor * sp.eye(adj.shape[0]))
    return tf.Tensor(adj, dtype=tf.float32)
