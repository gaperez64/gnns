import tensorflow as tf

from .base import Base
from .utils import glorot, normdAdjId,\
                   sparseDropout, sparseTensorFromMatrix,\
                   zeros


class GCN(Base):
    def __init__(self,
                 in_dim,               # input dimension
                 out_dim,              # output dimension
                 nonzero_feat_shape,   # shape of nonzero features
                 graph,                # graph to get the norm'd adj. matrix
                 labels,               # labels to measure losses and accuracy
                 labels_mask,          # mask for labels
                 hidden_layer_dim=16,  # hidden layer dimension
                 dropout_rate=0.5,     # dropout rate for training
                 weight_decay=5e-4):   # weight decay for parameter fitting
        super(GCN, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonzero_feat_shape = nonzero_feat_shape
        self.convolution_matrix =\
            sparseTensorFromMatrix(normdAdjId(graph))
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

    def call(self, x, training=None):
        # Layer 1: inputs are sparse
        x = sparseTensorFromMatrix(x)
        # if we are training, we enable the dropout layer
        if training:
            x = sparseDropout(x, self.dropout_rate,
                              self.nonzero_feat_shape)
        x = tf.sparse.sparse_dense_matmul(x, self.weight_matrix1)
        x = tf.sparse.sparse_dense_matmul(self.convolution_matrix, x)
        x = tf.nn.relu(x)
        # Setting up losses for training
        self.add_loss(self.weight_decay * tf.nn.l2_loss(self.weight_matrix1))

        # Layer 2: inputs are not sparse anymore and
        # for some reason we do not relu at the end
        if training:
            x = tf.nn.dropout(x, 1 - self.dropout_rate)
        x = tf.matmul(x, self.weight_matrix2)
        x = tf.sparse.sparse_dense_matmul(self.convolution_matrix, x)
        # Setting up losses for training
        self.add_loss(self.maskedSoftmaxXEntropy(x))

        # Add accuracy as a metric
        self.add_metric(self.maskedAccuracy(x), name="accuracy")

        return x


class GCN1(Base):
    def __init__(self,
                 in_dim,               # input dimension
                 out_dim,              # output dimension
                 nonzero_feat_shape,   # shape of nonzero features
                 graph,                # graph to get the norm'd adj. matrix
                 labels,               # labels to measure losses and accuracy
                 labels_mask,          # mask for labels
                 hidden_layer_dim=16,  # hidden layer dimension
                 dropout_rate=0.5,     # dropout rate for training
                 weight_decay=5e-4):   # weight decay for parameter fitting
        super(GCN1, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonzero_feat_shape = nonzero_feat_shape
        self.convolution_matrix =\
            sparseTensorFromMatrix(normdAdjId(graph))
        self.labels = labels
        self.labels_mask = labels_mask
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.weight_matrix1 = glorot(shape=[in_dim,
                                            out_dim],
                                     name="weights1")

    def call(self, x, training=None):
        # Layer 1: inputs are sparse
        x = sparseTensorFromMatrix(x)
        # if we are training, we enable the dropout layer
        if training:
            x = sparseDropout(x, self.dropout_rate,
                              self.nonzero_feat_shape)
        x = tf.sparse.sparse_dense_matmul(x, self.weight_matrix1)
        x = tf.sparse.sparse_dense_matmul(self.convolution_matrix, x)
        x = tf.nn.relu(x)
        # Setting up losses for training
        self.add_loss(self.weight_decay * tf.nn.l2_loss(self.weight_matrix1))
        self.add_loss(self.maskedSoftmaxXEntropy(x))

        # Add accuracy as a metric
        self.add_metric(self.maskedAccuracy(x), name="accuracy")

        return x


class GCN1p(Base):
    def __init__(self,
                 in_dim,               # input dimension
                 out_dim,              # output dimension
                 nonzero_feat_shape,   # shape of nonzero features
                 graph,                # graph to get the norm'd adj. matrix
                 labels,               # labels to measure losses and accuracy
                 labels_mask,          # mask for labels
                 hidden_layer_dim=16,  # hidden layer dimension
                 dropout_rate=0.5,     # dropout rate for training
                 weight_decay=5e-4,    # weight decay for parameter fitting
                 id_factor=0.9997):    # scaling factor a, for aI in D(A+aI)D
        super(GCN1p, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonzero_feat_shape = nonzero_feat_shape
        self.convolution_matrix =\
            sparseTensorFromMatrix(normdAdjId(graph,
                                              scaling_factor=id_factor))
        self.labels = labels
        self.labels_mask = labels_mask
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.weight_matrix1 = glorot(shape=[in_dim,
                                            out_dim],
                                     name="weights1")

    def call(self, x, training=None):
        # Layer 1: inputs are sparse
        x = sparseTensorFromMatrix(x)
        # if we are training, we enable the dropout layer
        if training:
            x = sparseDropout(x, self.dropout_rate,
                              self.nonzero_feat_shape)
        x = tf.sparse.sparse_dense_matmul(x, self.weight_matrix1)
        x = tf.sparse.sparse_dense_matmul(self.convolution_matrix, x)
        x = tf.nn.relu(x)
        # Setting up losses for training
        self.add_loss(self.weight_decay * tf.nn.l2_loss(self.weight_matrix1))
        self.add_loss(self.maskedSoftmaxXEntropy(x))

        # Add accuracy as a metric
        self.add_metric(self.maskedAccuracy(x), name="accuracy")

        return x


class GCN2p(Base):
    def __init__(self,
                 in_dim,               # input dimension
                 out_dim,              # output dimension
                 nonzero_feat_shape,   # shape of nonzero features
                 graph,                # graph to get the norm'd adj. matrix
                 labels,               # labels to measure losses and accuracy
                 labels_mask,          # mask for labels
                 hidden_layer_dim=16,  # hidden layer dimension
                 dropout_rate=0.5,     # dropout rate for training
                 weight_decay=5e-4,    # weight decay for parameter fitting
                 id_factor=0.9997):    # scaling factor a, for aI in D(A+aI)D
        super(GCN2p, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonzero_feat_shape = nonzero_feat_shape
        self.convolution_matrix =\
            sparseTensorFromMatrix(normdAdjId(graph,
                                              scaling_factor=id_factor))
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

    def call(self, x, training=None):
        # Layer 1: inputs are sparse
        x = sparseTensorFromMatrix(x)
        # if we are training, we enable the dropout layer
        if training:
            x = sparseDropout(x, self.dropout_rate,
                              self.nonzero_feat_shape)
        x = tf.sparse.sparse_dense_matmul(x, self.weight_matrix1)
        x = tf.sparse.sparse_dense_matmul(self.convolution_matrix, x)
        x = tf.nn.relu(x)
        # Setting up losses for training
        self.add_loss(self.weight_decay * tf.nn.l2_loss(self.weight_matrix1))

        # Layer 2: inputs are not sparse anymore
        if training:
            x = tf.nn.dropout(x, 1 - self.dropout_rate)
        x = tf.matmul(x, self.weight_matrix2)
        x = tf.sparse.sparse_dense_matmul(self.convolution_matrix, x)
        x = tf.nn.relu(x)
        # Setting up losses for training
        self.add_loss(self.weight_decay * tf.nn.l2_loss(self.weight_matrix2))
        self.add_loss(self.maskedSoftmaxXEntropy(x))

        # Add accuracy as a metric
        self.add_metric(self.maskedAccuracy(x), name="accuracy")

        return x


class GCN2pBias(Base):
    def __init__(self,
                 in_dim,               # input dimension
                 out_dim,              # output dimension
                 nonzero_feat_shape,   # shape of nonzero features
                 graph,                # graph to get the norm'd adj. matrix
                 labels,               # labels to measure losses and accuracy
                 labels_mask,          # mask for labels
                 hidden_layer_dim=16,  # hidden layer dimension
                 dropout_rate=0.5,     # dropout rate for training
                 weight_decay=5e-4,    # weight decay for parameter fitting
                 id_factor=0.9997):    # scaling factor a, for aI in D(A+aI)D
        super(GCN2pBias, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonzero_feat_shape = nonzero_feat_shape
        self.convolution_matrix =\
            sparseTensorFromMatrix(normdAdjId(graph,
                                              scaling_factor=id_factor))
        self.labels = labels
        self.labels_mask = labels_mask
        self.hidden_layer_dim = hidden_layer_dim
        self.dropout_rate = dropout_rate
        self.weight_decay = weight_decay
        self.weight_matrix1 = glorot(shape=[in_dim,
                                            hidden_layer_dim],
                                     name="weights1")
        self.bias_vector1 = zeros([hidden_layer_dim],
                                  name="bias1")
        self.weight_matrix2 = glorot(shape=[hidden_layer_dim,
                                            out_dim],
                                     name="weights2")
        self.bias_vector2 = zeros([out_dim], name="bias2")

    def call(self, x, training=None):
        # Layer 1: inputs are sparse
        x = sparseTensorFromMatrix(x)
        # if we are training, we enable the dropout layer
        if training:
            x = sparseDropout(x, self.dropout_rate,
                              self.nonzero_feat_shape)
        x = tf.sparse.sparse_dense_matmul(x, self.weight_matrix1)
        x = tf.sparse.sparse_dense_matmul(self.convolution_matrix, x)
        x = x + self.bias_vector1
        x = tf.nn.relu(x)
        # Setting up losses for training
        self.add_loss(self.weight_decay * tf.nn.l2_loss(self.weight_matrix1))
        self.add_loss(self.weight_decay * tf.nn.l2_loss(self.bias_vector1))

        # Layer 2: inputs are not sparse anymore
        if training:
            x = tf.nn.dropout(x, 1 - self.dropout_rate)
        x = tf.matmul(x, self.weight_matrix2)
        x = tf.sparse.sparse_dense_matmul(self.convolution_matrix, x)
        x = x + self.bias_vector2
        x = tf.nn.relu(x)
        # Setting up losses for training
        self.add_loss(self.weight_decay * tf.nn.l2_loss(self.weight_matrix2))
        self.add_loss(self.weight_decay * tf.nn.l2_loss(self.bias_vector2))
        self.add_loss(self.maskedSoftmaxXEntropy(x))

        # Add accuracy as a metric
        self.add_metric(self.maskedAccuracy(x), name="accuracy")

        return x
