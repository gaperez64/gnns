import numpy as np
import tensorflow as tf

from data.dataset import Dataset


# For reproducibility, we fix the random seed
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# We load the dataset
ds = Dataset("cora")

# We create the constant part of the model
#dad = tf.constant(ds.normd_adj_plus_id())
