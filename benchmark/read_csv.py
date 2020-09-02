"""
Reads csv benchmarks provided as input. Creates a DataFrame for each benchmark, and prints the table for easy reading.
The benchmarks are stored in the object benches

Depends:           pandas module (pip install pandas)

How to Run:        python -i read_csv.py /tmp/bench1.bench /tmp/bench2.bench ...
                   e.g.: ipython -i read_csv.py /tmp/Anymal_19DoF.bench /tmp/HyQ_19DoF.bench
                   e.g.: ipython -i read_csv.py /tmp/*.bench


Output:            benches
                   benches.bench1 (DataFrame), benches.bench2 (DataFrame)...

DataFrame API:     value = benches.bench1.loc[fn_name, nthreads, with(out)_cg][bench_parameter]
                   example:
                   mean_calc_time_withcg_with3threads = benches.bench1.loc["calc",3,True]["mean"]
                   max_calc_time_withoutcg_with2threads = benches.bench1.loc["calc",2,False]["max"]

"""

from __future__ import print_function
import pandas as pd
import sys
from os.path import exists, splitext, basename
pd.options.display.width = 0


class Benchmarks():
    pass


def parseCsvFile(filename, sep=',', delimiter=None):
    """Inputs a file and ouputs a matrix.
    input: (string) filename
    output: (numpy array) seq
    """
    col_types = dict(fn_name=str,
                     nthreads=int,
                     with_cg=bool,
                     mean=float,
                     stddev=float,
                     max=float,
                     min=float,
                     mean_per_nodes=float,
                     stddev_per_nodes=float)

    seq = pd.read_csv(filename,
                      header=0,
                      sep=sep,
                      delim_whitespace=False,
                      quoting=2,
                      index_col=[0, 1, 2],
                      dtype=col_types)

    return seq


benches = Benchmarks()

filenames = sys.argv[1:]
for filename in filenames:
    if not exists(filename):
        continue
    benchname = splitext(basename(filename))[0]
    seq = parseCsvFile(filename)
    setattr(benches, benchname, seq)
    print(benchname)
    print(seq)
    print("********************")
