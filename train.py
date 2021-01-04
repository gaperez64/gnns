import matplotlib.pyplot as plt
import numpy as np
import sys
import tensorflow as tf

from data.dataset import Dataset
import models.kipf


def trainModel(model, dataset, num_epochs=200, debug=False):
    # Train the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    val_acc_list = []
    for epoch in range(num_epochs):
        # we open a GradientTape to record the operations run during
        # the forward pass of the input data through the model,
        # this enables automatic differentiation
        with tf.GradientTape() as tape:
            # run the forward pass
            _ = model(dataset.features, training=True)
            # compute the loss value
            train_loss = sum(model.losses)
            # and the accuracy
            train_acc = model.metrics[0].result()
        # use the GradientTape to automatically retrieve the gradients
        # of the trainable weights w.r.t. the losses
        grads = tape.gradient(train_loss, model.trainable_weights)
        # run one step of gradient descent
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # we also run a validation forward pass to see how we are doing w.r.t.
        # to the validation part of the data set
        model.setLabels(dataset.labels_val, dataset.val_mask)
        _ = model(dataset.features)
        val_loss = sum(model.losses)
        val_acc = model.metrics[0].result()
        if debug:
            print(f"Epoch {epoch + 1:04d} " +
                  f"train loss={float(train_loss):.5f}, " +
                  f"train acc={float(train_acc):.5f}; " +
                  f"valid loss={float(val_loss):.5f}, " +
                  f"valid acc={float(val_acc):.5f}")
        val_acc_list.append(val_acc)

        # we reset the labels to the training ones
        model.setLabels(dataset.labels_train, dataset.train_mask)
    if debug:
        print("=== Training finished ===")
    # to conclude, we run a test forward pass
    model.setLabels(dataset.labels_test, dataset.test_mask)
    _ = model(dataset.features)
    test_loss = sum(model.losses)
    test_acc = model.metrics[0].result()
    if debug:
        print(f"test loss={float(test_loss):.5f}, " +
              f"test acc={float(test_acc):.5f}")

    return (test_loss, test_acc, val_acc_list)


def main(plot=False):
    # For reproducibility, we fix the random seed
    seed = 123
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # We load the dataset
    gnns = {"gcn": models.kipf.GCN,
            "1layr-gcn": models.kipf.GCN1,
            "1layr-p-gcn": models.kipf.GCN1p,
            "2layr-p-gcn": models.kipf.GCN2p}
    for dsname in ["cora", "citeseer", "pubmed"]:
        ds = Dataset(dsname)
        for name, GNN in gnns.items():
            # We instantiate a gcn model
            gcn = GNN(in_dim=ds.features.shape[1],
                      out_dim=ds.labels_train.shape[1],
                      nonzero_feat_shape=ds.features.data.shape,
                      graph=ds.graph,
                      labels=ds.labels_train,
                      labels_mask=ds.train_mask)
            # Train the gcn
            (loss, acc, val_accs) = trainModel(gcn, ds)
            print(f"== Trained GNN {name} on dataset {dsname} ==")
            print(f"test loss={float(loss):.5f}, " +
                  f"test acc={float(acc):.5f}")
            if plot:
                plt.plot(range(len(val_accs)),
                         val_accs, label=f"{name}")
        if plot:
            plt.legend(loc="lower right")
            plt.title(f"\"{dsname}\" dataset")
            plt.xlabel("Training epochs")
            plt.ylabel("Validation accuracy")
            plt.savefig(dsname)
            # plt.show()


if __name__ == "__main__":
    if len(sys.argv) > 2 or len(sys.argv) == 2 and sys.argv[1] != "-plot":
        print(f"{sys.argv[0]} expects at most one option \"-plot\":\n" +
              "whether to plot valuation values", file=sys.stderr)
        exit(1)
    main(len(sys.argv) == 2)
    exit(0)
