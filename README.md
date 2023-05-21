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
as well as their respective dependencies (see `requirements.txt`)

# Data Preparation
Statistics regarding the considered benchmarks:
* min no. of latches = 0
* mean no. of latches = 510.95
* max no. of latches = 43950
* No. of distinct latch names for the whole data set = 144043
This information suggests we cluster the latch names to get a set of feature
to use in our learning task.


# Training Setup

