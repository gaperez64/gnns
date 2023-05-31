#!/usr/bin/env python3

"""
This script is based on Frames Catherine White's answer to the question

https://stats.stackexchange.com/questions/123060/clustering-a-long-list-of-strings-words-into-similarity-groups

on how to cluster strings into similarity groups.
"""

from functools import lru_cache
import numpy as np
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering
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
    word_list = [latch.strip() for latch in latchNames]
    latchNames.close()
    print(f"Number of latch names = {len(word_list)}")

    # some manual transformations
    # lo = re.compile("lo\\d*")
    # latch = re.compile("latch\\d*")
    # label = re.compile("label__l\\d*")
    # signal = re.compile("signal\\d*")
    # words = []
    # for word in word_list:
    #     if lo.match(word) is not None:
    #         words.append("loX")
    #     elif latch.match(word) is not None:
    #         words.append("latchX")
    #     elif label.match(word) is not None:
    #         words.append("label__lX")
    #     elif signal.match(word) is not None:
    #         words.append("signalX")
    #     else:
    #         words.append(word)
    words = word_list[:1000]
    words = sorted(list(set(words)), key=lambda w: len(w))
    print(f"After manual treatment = {len(words)}")
    return words


def prepLevMatrix(words):
    # prepare levenshtein distance matrix
    print("=== Computing levenshtein distances ===")
    lev_list = []
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
    np.save("lev_array.npy", lev_array)
    print("=== Done computing distances ===")
    return clusterNames(words, lev_array)


def clusterNames(words, lev_array):
    words = np.asarray(words)  # so that it can be indexed with arrays
    lev_similarity = -1.0 * lev_array
    # cluster_algo = AgglomerativeClustering(n_clusters=50,
    #                                        metric='precomputed',
    #                                        linkage='average')
    cluster_algo = AffinityPropagation(affinity="precomputed", damping=0.5,
                                       verbose=True, random_state=0)
    clusters = cluster_algo.fit_predict(lev_similarity)
    cluster_strings = []
    for cid in np.unique(clusters):
        c = words[np.nonzero(clusters == cid)]
        cstring = ", ".join(c)
        print(f"Cluster {cid}: {cstring}")
        cluster_strings.append(cstring)
    return cluster_strings


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
        clusters = prepLevMatrix(latchNames)
        printClusters(clusters, sys.argv[2])
        exit(0)
    else:
        latchNames = latchNamesFromFile(sys.argv[1])
        lev_array = np.load(sys.argv[3])
        clusters = clusterNames(latchNames, lev_array)
        printClusters(clusters, sys.argv[2])
        exit(0)
