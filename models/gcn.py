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
    return tf.Variable(initial, name=name, dtype=tf.float32,
                       trainable=True)


class GCN(tf.keras.Model):
    def __init__(self, in_dim, out_dim,
                 nonzero_feat_shape,
                 convolution_matrix,
                 labels, labels_mask,
                 hidden_layer_dim=16,
                 dropout_rate=0.5,
                 weight_decay=5e-4):
        super(GCN, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonzero_feat_shape = nonzero_feat_shape
        self.convolution_matrix = sparseTensorFromMatrix(convolution_matrix)
        self.labels = labels
        self.labels_mask = labels_mask
        self.hidden_layer_dim = hidden_layer_dim
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.weight_matrix1 = glorot(shape=[in_dim,
                                            hidden_layer_dim],
                                     name="weights1")
        self.weight_matrix2 = glorot(shape=[hidden_layer_dim,
                                            out_dim],
                                     name="weights2")

    def maskedSoftmaxXEntropy(self, preds):
        """Softmax cross-entropy loss with masking"""
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds,
                                                       labels=self.labels)
        mask = tf.cast(self.labels_mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss *= mask
        return tf.reduce_mean(loss)

    def maskedAccuracy(self, preds):
        """Accuracy with masking"""
        correct_prediction = tf.equal(tf.argmax(preds, 1),
                                      tf.argmax(self.labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(self.labels_mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def call(self, x):
        # Layer 1: inputs are sparse
        x = sparseTensorFromMatrix(x)
        x = sparseDropout(x, self.dropout_rate,
                          self.nonzero_feat_shape)
        x = tf.sparse.sparse_dense_matmul(x, self.weight_matrix1)
        x = tf.sparse.sparse_dense_matmul(self.convolution_matrix, x)
        x = tf.nn.relu(x)
        # Setting up losses for training
        self.add_loss(self.weight_decay * tf.nn.l2_loss(self.weight_matrix1))

        # Layer 2: inputs are not sparse anymore and
        # for some reason we do no relu
        x = tf.nn.dropout(x, 1 - self.dropout_rate)
        x = tf.matmul(x, self.weight_matrix2)
        x = tf.sparse.sparse_dense_matmul(self.convolution_matrix, x)
        # Setting up losses for training
        self.add_loss(self.maskedSoftmaxXEntropy(x))

        # Add accuracy as a metric
        self.add_metric(self.maskedAccuracy(x), name="accuracy")

        return x
