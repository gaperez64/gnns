#!/usr/bin/env python3

"""
This script is based on Frames Catherine White's answer to the question

https://stats.stackexchange.com/questions/123060/clustering-a-long-list-of-strings-words-into-similarity-groups

on how to cluster strings into similarity groups.
"""

from distance import levenshtein
from functools import lru_cache
import numpy as np
from sklearn.cluster import AffinityPropagation
import sys


def clusterNamesFromFile(fname):
    latchNames = open(fname, "r")
    words = np.asarray([latch for latch in latchNames])
    latchNames.close()
    print("Computing levenshtein distances")

    @lru_cache()
    def _lev(x, y):
        levenshtein(x, y)

    lev_list = [[_lev(words[i], words[j]) if i <= j else
                 _lev(words[j], words[i]) for i in range(len(words))]
                for j in range(len(words))]
    lev_array = np.array(lev_list)
    print("Done! saving them now")
    np.save(lev_array, "lev_array.npy")
    return clusterNames(lev_array)


def clusterNames(words, lev_array):
    lev_similarity = -1 * lev_array
    affprop = AffinityPropagation(affinity="precomputed",
                                  damping=0.5).fit(lev_similarity)
    for cluster_id in np.unique(affprop.labels_):
        exemplar = words[affprop.cluster_centers_indices_[cluster_id]]
        cluster = np.unique(words[np.nonzero(affprop.labels_ == cluster_id)])
        cluster_str = ", ".join(cluster)
        print(" - *%s:* %s" % (exemplar, cluster_str))
    return words[affprop.cluster_centers_indices_]


def printClusters(latchNames, fname):
    out = open(fname, "w")
    for latch in latchNames:
        out.write(f"{latch}\n")
    out.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Two positional arguments expected: "
              "(1) the full path of the file with the latch names"
              "(2) the full path of the file where you want the output",
              file=sys.stderr)
        exit(1)
    else:
        latchNames = clusterNamesFromFile(sys.argv[1])
        printClusters(latchNames, sys.argv[2])
        exit(0)
