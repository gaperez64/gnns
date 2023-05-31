#!/usr/bin/env python3

"""
This script is based on Frames Catherine White's answer to the question

https://stats.stackexchange.com/questions/123060/clustering-a-long-list-of-strings-words-into-similarity-groups

on how to cluster strings into similarity groups.
"""

from functools import lru_cache
import numpy as np
from sklearn.cluster import AffinityPropagation
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


def latchNamesFromFile(fname):
    latchNames = open(fname, "r")
    words = [latch.strip() for latch in latchNames]
    latchNames.close()
    print(f"Number of latch names = {len(words)}")
    return words


def levMatAndCluster(words):
    # prepare levenshtein distance matrix
    # print("=== Computing levenshtein distances ===")
    lev_list = []
    for i in range(len(words)):
        lev_list.append([])
        # if i % 10 == 0:
        #     print(f"Treating word {i + 1}: {words[i]}")
        for j in range(len(words)):
            if i == j:
                lev_list[i].append(0)
            elif i <= j:
                lev_list[i].append(levenshtein(words[i], words[j]))
            else:
                lev_list[i].append(levenshtein(words[j], words[i]))
    lev_array = np.array(lev_list)
    np.save("lev_array.npy", lev_array)
    # print("=== Done computing distances ===")
    return clusterNames(words, lev_array)


def clusterNames(words, lev_array):
    words = np.asarray(words)  # so that it can be indexed with arrays
    lev_similarity = -1.0 * lev_array
    cluster_algo = AffinityPropagation(affinity="precomputed", damping=0.5,
                                       verbose=True, random_state=0)
    cluster_algo.fit_predict(lev_similarity)
    unique_labels = np.unique(cluster_algo.labels_)
    c0 = words[cluster_algo.cluster_centers_indices_[unique_labels]]
    return c0


def printClusters(latchNames, fname):
    out = open(fname, "w")
    for latch in latchNames:
        out.write(f"{latch}\n")
    out.close()


if __name__ == "__main__":
    if len(sys.argv) not in [3, 4]:
        print("Three positional arguments expected:\n"
              "(1) the full path of the file with the latch names\n"
              "(2) the full path of the file where you want the output\n"
              "(3) optional: distance matrix file",
              file=sys.stderr)
        exit(1)
    elif len(sys.argv) == 3:
        latchNames = latchNamesFromFile(sys.argv[1])
        # partition into sub lists of latch names
        sublists = [latchNames]
        changed = True
        while changed:
            changed = False
            newlists = []
            for sub in sublists:
                if len(sub) > 1000:  # FIXME 1K hardcoded
                    newlists.append(sub[len(sub) // 2:])
                    newlists.append(sub[:len(sub) // 2:])
                    changed = True
                else:
                    newlists.append(sub)
            sublists = newlists
        # cluster all sublists and then merge them
        newlists = []
        print(f"No. of sublists = {len(sublists)}")
        print([len(sub) for sub in sublists])
        for sub in sublists:
            clusters = levMatAndCluster(sub)
            print(f"No. of resulting clusters = {len(clusters)}")
            newlists.extend(clusters)
        # do a final clustering
        clusters = levMatAndCluster(sub)
        print(f"Final no. of clusters = {len(clusters)}")
        printClusters(clusters, sys.argv[2])
        exit(0)
    else:
        latchNames = latchNamesFromFile(sys.argv[1])
        lev_array = np.load(sys.argv[3])
        clusters = clusterNames(latchNames, lev_array)
        printClusters(clusters, sys.argv[2])
        exit(0)
