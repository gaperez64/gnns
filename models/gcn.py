import tensorflow as tf

from .utils import glorot, sparseDropout, sparseTensorFromMatrix


class GCN(tf.keras.Model):
    def __init__(self,
                 in_dim,               # input dimension
                 out_dim,              # output dimension
                 nonzero_feat_shape,   # shape of nonzero features
                 convolution_matrix,   # the normalized adjacency matrix
                 labels,               # labels to measure losses and accuracy
                 labels_mask,          # mask for labels
                 hidden_layer_dim=16,  # hidden layer dimension
                 dropout_rate=0.5,     # dropout rate for training
                 weight_decay=5e-4):   # weight decay for parameter fitting
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

    def setLabels(self, labels, labels_mask):
        self.labels = labels
        self.labels_mask = labels_mask

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
