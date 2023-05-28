#!/usr/bin/env python3

"""
This script is based on Frames Catherine White's answer to the question

https://stats.stackexchange.com/questions/123060/clustering-a-long-list-of-strings-words-into-similarity-groups

on how to cluster strings into similarity groups.
"""

from functools import lru_cache
import numpy as np
from sklearn.cluster import KMeans
import sys


@lru_cache()
def levenshtein(x, y):
    # previous row of distances, i.e. distances from
    # an empty x to y of length i
    prev = [i for i in range(0, len(y) + 1)]
    for i in range(0, len(x)):
        # compute the current row distances, i.e. compare
        # distances from x of length i + 1 to empty y
        cur = []
        cur.append(i + 1)
        # fill in the rest inductively
        for j in range(0, len(y)):
            delCost = prev[j + 1] + 1
            insCost = prev[j] + 1
            subCost = cur[j] if x[i] == y[j] else cur[j] + 1
            cur.append(min([delCost, insCost, subCost]))
        prev = cur
    return prev[-1]


def clusterNamesFromFile(fname):
    latchNames = open(fname, "r")
    words = np.asarray([latch.strip() for latch in latchNames])
    latchNames.close()
    lev_list = []
    mlen = max([len(w) for w in words])
    print(f"Number of latch names = {len(words)}")
    print(f"Max length of latch names = {mlen}")

    # partition list by lengths
    words = sorted(words, key=lambda w: len(w))
    curlen = len(words[0])
    partlen = {curlen: []}
    idx = 0
    for w in words:
        if len(w) > curlen:
            curlen = len(w)
            partlen[curlen] = []
        partlen[curlen].append((w, idx))
        idx += 1
    print("Strings grouped by length:")
    print({k: len(v) for k, v in partlen.items()})

    # Start of levenshtein computations
    print("=== Computing levenshtein distances ===")
    for i in range(len(words)):
        lev_list.append([])
        if i % 10 == 0:
            print(f"Treating word {i + 1}: {words[i]}")
        for j in range(len(words)):
            if i == j:
                lev_list[i].append(0)
            elif i <= j:
                lev_list[i].append(levenshtein(words[i], words[j]))
            else:
                lev_list[i].append(levenshtein(words[j], words[i]))
    lev_array = np.array(lev_list)
    print("Done! saving them now")
    np.save(lev_array, "lev_array.npy")
    return clusterNames(lev_array)


def clusterNames(words, lev_array):
    lev_similarity = -1 * lev_array
    affprop = KMeans(n_clusters=300,
                     random_state=0,  # to make deterministic
                     n_init="auto").fit(lev_similarity)
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
        print("Two positional arguments expected:\n"
              "(1) the full path of the file with the latch names\n"
              "(2) the full path of the file where you want the output",
              file=sys.stderr)
        exit(1)
    else:
        latchNames = clusterNamesFromFile(sys.argv[1])
        printClusters(latchNames, sys.argv[2])
        exit(0)
