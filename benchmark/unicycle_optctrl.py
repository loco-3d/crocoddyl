import os
import subprocess
import sys
import time

import crocoddyl
import numpy as np
from crocoddyl.utils import UnicycleModelDerived

N = 200  # number of nodes
T = int(sys.argv[1]) if (len(sys.argv) > 1) else int(5e3)  # number of trials
MAXITER = 1
CALLBACKS = False


def createProblem(model):
    x0 = np.matrix([[1.0], [0.0], [0.0]])
    runningModels = [model()] * N
    terminalModel = model()
    xs = [x0] * (N + 1)
    us = [np.matrix([[0.0], [0.0]])] * N

    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    return xs, us, problem


def runDDPSolveBenchmark(xs, us, problem):
    ddp = crocoddyl.SolverDDP(problem)
    if CALLBACKS:
        ddp.setCallbacks([crocoddyl.CallbackVerbose()])
    duration = []
    for i in range(T):
        c_start = time.time()
        ddp.solve(xs, us, MAXITER)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_dur = sum(duration) / len(duration)
    min_dur = min(duration)
    max_dur = max(duration)
    return avrg_dur, min_dur, max_dur


def runShootingProblemCalcBenchmark(xs, us, problem):
    duration = []
    for i in range(T):
        c_start = time.time()
        problem.calc(xs, us)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_dur = sum(duration) / len(duration)
    min_dur = min(duration)
    max_dur = max(duration)
    return avrg_dur, min_dur, max_dur


def runShootingProblemCalcDiffBenchmark(xs, us, problem):
    duration = []
    for i in range(T):
        c_start = time.time()
        problem.calcDiff(xs, us)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_dur = sum(duration) / len(duration)
    min_dur = min(duration)
    max_dur = max(duration)
    return avrg_dur, min_dur, max_dur


print("\033[1m")
print("C++:")
popen = subprocess.check_call(
    [os.path.dirname(os.path.abspath(__file__)) + "/unicycle-optctrl", str(T)]
)

print("Python bindings:")
xs, us, problem = createProblem(crocoddyl.ActionModelUnicycle)
avrg_dur, min_dur, max_dur = runDDPSolveBenchmark(xs, us, problem)
print(f"  DDP.solve [ms]: {avrg_dur} ({min_dur}, {max_dur})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcBenchmark(xs, us, problem)
print(f"  ShootingProblem.calc [ms]: {avrg_dur} ({min_dur}, {max_dur})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcDiffBenchmark(xs, us, problem)
print(f"  ShootingProblem.calcDiff [ms]: {avrg_dur} ({min_dur}, {max_dur})")

print("Python:")
xs, us, problem = createProblem(UnicycleModelDerived)
avrg_dur, min_dur, max_dur = runDDPSolveBenchmark(xs, us, problem)
print(f"  DDP.solve [ms]: {avrg_dur} ({min_dur}, {max_dur})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcBenchmark(xs, us, problem)
print(f"  ShootingProblem.calc [ms]: {avrg_dur} ({min_dur}, {max_dur})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcDiffBenchmark(xs, us, problem)
print(f"  ShootingProblem.calcDiff [ms]: {avrg_dur} ({min_dur}, {max_dur})")
print("\033[0m")
