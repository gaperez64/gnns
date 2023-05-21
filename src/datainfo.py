#!/usr/bin/env python3

import aiger
import os
import sys

from statistics import mean


def scanBenchmarks(inputRoot):
    latchNames = set()
    skipped = []
    stats = []
    for d in os.walk(inputRoot):
        root = d[0]
        for fileName in os.listdir(root):
            if fileName.endswith(".aag"):
                fullName = os.path.join(root, fileName)
                try:
                    aig = aiger.load(fullName)
                except Exception as err:
                    skipped.append(tuple([fullName, err]))
                latches = aig.latches
                stats.append(len(latches))
                latchNames.update([s.lower() for s in latches])
    for (f, e) in skipped:
        print(f"Warning, skipped benchmark {fullName}", file=sys.stderr)
        # print(err)
    print(f"min={min(stats)}, mean={mean(stats)}, max={max(stats)}")
    return frozenset(latchNames)


def printLatchNames(latchNames, fname):
    out = open(fname, "w")
    for latch in latchNames:
        out.write(f"{latch}\n")
    out.close()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Two positional arguments expected: "
              "(1) the path to the root of the benchmark directories"
              "(2) the full path of the file where you want latch names",
              file=sys.stderr)
        exit(1)
    else:
        latchNames = scanBenchmarks(sys.argv[1])
        printLatchNames(latchNames, sys.argv[2])
        exit(0)
