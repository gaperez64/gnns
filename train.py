import numpy as np
import tensorflow as tf
import scipy.sparse as sp

from data.dataset import Dataset
from models.gcn import GCN


# For reproducibility, we fix the random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# We load the dataset
ds = Dataset("cora")

# We instantiate a gcn model
gcn = GCN(in_dim=ds.features.shape[1],
          out_dim=ds.labels_train.shape[1],
          nonzero_feat_shape=ds.features.data.shape,
          convolution_matrix=ds.normdAdjId(),
          labels=ds.labels_train,
          labels_mask=ds.train_mask)

# We print some debug info
print(f"First output = {gcn(ds.features)}")
print(gcn.variables)
print(gcn.summary())
assert(not sp.isspmatrix_coo(ds.features))
sparse_mx = ds.features.tocoo()
coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
values = sparse_mx.data
shape = sparse_mx.shape
print(f"Feature coords = {coords}")
print(f"Feature values = {values}")
print(f"Feature shape = {shape}")


# Train the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
for epoch in range(1, 11):
    # we open a GradientTape to record the operations run during
    # the forward pass of the input data through the model,
    # this enables automatic differentiation
    with tf.GradientTape() as tape:
        # run the forward pass
        logits = gcn(ds.features, training=True)
        # compute the loss value
        loss_val = sum(gcn.losses)
        # and the accuracy
        acc_val = gcn.metrics[0].result()
    # use the GradientTape to automatically retrieve the gradients
    # of the trainable weights w.r.t. the losses
    grads = tape.gradient(loss_val, gcn.trainable_weights)
    # run one step of gradient descent
    optimizer.apply_gradients(zip(grads, gcn.trainable_weights))
    print(f"Epoch {epoch:04d} train loss={float(loss_val):.5f}, " +
          f"train acc={float(acc_val):.5f}")
    print(f"logits = {logits}")
