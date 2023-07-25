# Graph Neural Networks for Bisimulation

We implement a GNN-based bisimulation computation for large graphs coming from
the Hardware Model Checking Competition (HWMCC) and the Reactive Synthesis
Competition (SYNTCOMP). All [benchmarks
considered](https://github.com/SYNTCOMP/benchmarks/tree/master/aiger) are in
the AIGER format.

## Dependencies
The following Python 3 libraries
* `tensorflow-gnn`
* `py-aiger`
* `distance`
* `scikit-learn`
as well as their respective dependencies

# Data Preparation

## Feature collection
The list of all latch names from the benchmarks under consideration was
generated using `src/datainfo.py` and stored in `data/latch_names.txt`.

Statistics regarding the considered benchmarks:
* min no. of latches = 0
* mean no. of latches = 510.95
* max no. of latches = 43950
* No. of distinct latch names for the whole data set = 144043
This information suggests we cluster the latch names to get a set of feature
to use in our learning task.

The script `src/clusterLatches.py` has been used to obtain 56 clusters of
latches stored in `data/latch_clusters.txt`.

## Labelled data collection
TODO

* Note: the main hindrance from here onward is the need for an adjacency
  matrix which (if kept entirely in memory) is too large; we can ignore this
  and start with the smaller benchmarks first
* Note: since we have DFAs, bisimulation equivalence implies language
  equivalence!

## GNN (architecture) creation
TODO

## Training setup
TODO
