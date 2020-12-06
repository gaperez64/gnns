import networkx
import numpy as np
import pickle
import scipy.sparse as sp


def parse_index_file(filename):
    """Parse index file"""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, no_rows):
    """Create mask"""
    mask = np.zeros(no_rows)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation"""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape
    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)
    return sparse_mx


class Dataset:
    """
    Includes all the information we use from a graph-based
    dataset
    """
    graph: networkx.Graph
    features: tuple
    labels_train: np.ndarray
    labels_eval: np.ndarray
    labels_test: np.ndarray
    train_mask: np.ndarray
    eval_mask: np.ndarray
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
            parse_index_file(f"{dataset_dir}/ind.{dataset_str}.test.index")
        test_idx_range = np.sort(test_idx_reorder)

        features = sp.vstack((allx, tx)).tolil()
        features[test_idx_reorder, :] = features[test_idx_range, :]

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y) + 500)

        train_mask = sample_mask(idx_train, labels.shape[0])
        val_mask = sample_mask(idx_val, labels.shape[0])
        test_mask = sample_mask(idx_test, labels.shape[0])

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
        features = sparse_to_tuple(r_mat_inv.dot(features))

        # We can now store the values of the object's attributes
        self.graph = networkx.from_dict_of_lists(graph)
        self.features = features
        self.labels_train = y_train
        self.labels_eval = y_val
        self.labels_test = y_test
        self.train_mask = train_mask
        self.eval_mask = val_mask
        self.test_mask = test_mask

    def normalized_adj(adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def normd_adj_plus_id(self):
        """Recover the symmetrically-normalized adjacency matrix
        with an identity added
        """
        adj = networkx.adjacency_matrix(self.graph)
        # adding the identity matrix
        adj = adj + sp.eye(adj.shape[0])
        adj = sp.coo_matrix(adj)
        # now that we have added the identity matrix, we can normalize
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        normd = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
        return sparse_to_tuple(normd.tocoo())
