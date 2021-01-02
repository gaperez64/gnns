import networkx
import numpy as np
import pickle
import scipy.sparse as sp


def parseIndexFile(filename):
    """Parse index file"""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sampleMask(idx, no_rows):
    """Create mask"""
    mask = np.zeros(no_rows)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


class Dataset:
    """
    Includes all the information we use from a graph-based
    dataset
    """
    graph: networkx.Graph
    features: np.ndarray
    labels_train: np.ndarray
    labels_val: np.ndarray
    labels_test: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray

    def __init__(self, dataset_str, dataset_dir="./res"):
        """
        Loads data from pickle files

        ind.dataset_str.x => the feature vectors of the training instances as
        scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.tx => the feature vectors of the test instances as
        scipy.sparse.csr.csr_matrix object; ind.dataset_str.allx => the
        feature vectors of both labeled and unlabeled
        training instances (a superset of ind.dataset_str.x) as
        scipy.sparse.csr.csr_matrix object;
        ind.dataset_str.y => the one-hot labels of the labeled training
        instances as numpy.ndarray object;
        ind.dataset_str.ty => the one-hot labels of the test instances as
        numpy.ndarray object;
        ind.dataset_str.ally => the labels for instances in
        ind.dataset_str.allx as numpy.ndarray object;
        ind.dataset_str.graph => a dict in the format {index:
        [index_of_neighbor_nodes]} as collections.defaultdict object;
        ind.dataset_str.test.index => the indices of test instances in graph,
        for the inductive setting as list object.

        :param dataset_str: Dataset name
        :param dataset_dir: Directory containing the pickle files
        :return: a Dataset object with all data
        """
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open(f"{dataset_dir}/ind.{dataset_str}.{names[i]}",
                      "rb") as f:
                objects.append(pickle.load(f))

        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder =\
            parseIndexFile(f"{dataset_dir}/ind.{dataset_str}.test.index")
        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        train_mask = sampleMask(idx_train, labels.shape[0])
        val_mask = sampleMask(idx_val, labels.shape[0])
        test_mask = sampleMask(idx_test, labels.shape[0])

        y_train = np.zeros(labels.shape)
        y_val = np.zeros(labels.shape)
        y_test = np.zeros(labels.shape)
        y_train[train_mask, :] = labels[train_mask, :]
        y_val[val_mask, :] = labels[val_mask, :]
        y_test[test_mask, :] = labels[test_mask, :]

        # Before we finish, we normalize the feature matrix
        # and convert it to tuple representation
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)

        # We can now store the values of the object's attributes
        self.graph = networkx.from_dict_of_lists(graph)
        self.features = features
        self.labels_train = y_train
        self.labels_val = y_val
        self.labels_test = y_test
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask
