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
          convolution_matrix=ds.normd_adj_plus_id())

# Test the model on an input
print(gcn(ds.features))
