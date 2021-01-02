import numpy as np
import tensorflow as tf

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

# Train the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
for epoch in range(200):
    # we open a GradientTape to record the operations run during
    # the forward pass of the input data through the model,
    # this enables automatic differentiation
    with tf.GradientTape() as tape:
        # run the forward pass
        logits = gcn(ds.features, training=True)
        # compute the loss value
        train_loss = sum(gcn.losses)
        # and the accuracy
        train_acc = gcn.metrics[0].result()
    # use the GradientTape to automatically retrieve the gradients
    # of the trainable weights w.r.t. the losses
    grads = tape.gradient(train_loss, gcn.trainable_weights)
    # run one step of gradient descent
    optimizer.apply_gradients(zip(grads, gcn.trainable_weights))

    # we also run a validation forward pass to see how we are doing w.r.t.
    # to the validation part of the data set
    gcn.setLabels(ds.labels_val, ds.val_mask)
    logits = gcn(ds.features)
    val_loss = sum(gcn.losses)
    val_acc = gcn.metrics[0].result()
    print(f"Epoch {epoch + 1:04d} train loss={float(train_loss):.5f}, " +
          f"train acc={float(train_acc):.5f}; " +
          f"valid loss={float(val_loss):.5f}, " +
          f"valid acc={float(val_acc):.5f}")

    # we reset the labels to the training ones
    gcn.setLabels(ds.labels_train, ds.train_mask)

print("=== Training finished ===")
# to conclude, we run a test forward pass
gcn.setLabels(ds.labels_test, ds.test_mask)
logits = gcn(ds.features)
test_loss = sum(gcn.losses)
test_acc = gcn.metrics[0].result()
print(f"test loss={float(test_loss):.5f}, " +
      f"test acc={float(test_acc):.5f}")
exit(0)
