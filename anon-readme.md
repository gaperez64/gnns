# Graph Neural Networks
This is a reimplementation of Kipf & Welling's graph convolutional networks in
Python 3 with Tensorflow 2. Additionally, several modifications and
alternative architectures are implemented for the purpose of comparison.

## Running the ICML-submission training setup
1. It is recommended that you use a virtual environment with Python3.8 (higher
   versions are not tensorflow 2.0 friendly). For your convenience, there is a
   requirements.txt included which can be used with pip install.
2. Running `python train.py' will launch all tests used for the submitted
   paper and will generate pdf-format matplotlib graphs. Additionally, the
   test results will be printed to the standard output.

## Datasets
We recycle Kipf & Welling's datasets. That is we use citation network data
(Cora, Citeseer, or Pubmed). The original datasets can be found here:
http://www.cs.umd.edu/~sen/lbc-proj/LBC.html. The citeseer dataset has been
"fixed" by Kipf due to isolated nodes. In the `res` directory we use dataset
splits provided by https://github.com/kimiyoung/planetoid and described in the
following article: 
```
@article{DBLP:journals/corr/YangCS16,
  author    = {Zhilin Yang and
               William W. Cohen and
               Ruslan Salakhutdinov},
  title     = {Revisiting Semi-Supervised Learning with Graph Embeddings},
  journal   = {CoRR},
  volume    = {abs/1603.08861},
  year      = {2016},
  url       = {http://arxiv.org/abs/1603.08861},
  archivePrefix = {arXiv},
  eprint    = {1603.08861},
  timestamp = {Mon, 13 Aug 2018 16:46:02 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/YangCS16.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

# References
If you use (part of) this project in some of your work, consider citing the
original paper by Kipf and Welling:
```
@inproceedings{DBLP:conf/iclr/KipfW17,
  author    = {Thomas N. Kipf and
               Max Welling},
  title     = {Semi-Supervised Classification with Graph Convolutional Networks},
  booktitle = {5th International Conference on Learning Representations, {ICLR} 2017,
               Toulon, France, April 24-26, 2017, Conference Track Proceedings},
  publisher = {OpenReview.net},
  year      = {2017},
  url       = {https://openreview.net/forum?id=SJU4ayYgl},
}
```
