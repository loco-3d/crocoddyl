import os
import subprocess
import sys
import time

import crocoddyl
import numpy as np
from crocoddyl.utils import LQRModelDerived

NX = 37
NU = 12
N = 100  # number of nodes
T = int(sys.argv[1]) if (len(sys.argv) > 1) else int(5e3)  # number of trials
MAXITER = 1
CALLBACKS = False


def createProblem(model):
    x0 = np.matrix(np.zeros(NX)).T
    runningModels = [model(NX, NU)] * N
    terminalModel = model(NX, NU)
    xs = [x0] * (N + 1)
    us = [np.matrix(np.zeros(NU)).T] * N

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

    avrg_duration = sum(duration) / len(duration)
    min_duration = min(duration)
    max_duration = max(duration)
    return avrg_duration, min_duration, max_duration


def runShootingProblemCalcBenchmark(xs, us, problem):
    duration = []
    for i in range(T):
        c_start = time.time()
        problem.calc(xs, us)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_duration = sum(duration) / len(duration)
    min_duration = min(duration)
    max_duration = max(duration)
    return avrg_duration, min_duration, max_duration


def runShootingProblemCalcDiffBenchmark(xs, us, problem):
    duration = []
    for i in range(T):
        c_start = time.time()
        problem.calcDiff(xs, us)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_duration = sum(duration) / len(duration)
    min_duration = min(duration)
    max_duration = max(duration)
    return avrg_duration, min_duration, max_duration


print("\033[1m")
print("C++:")
popen = subprocess.check_call(
    [os.path.dirname(os.path.abspath(__file__)) + "/lqr-optctrl", str(T)]
)

print("Python bindings:")
xs, us, problem = createProblem(crocoddyl.ActionModelLQR)
avrg_dur, min_dur, max_dur = runDDPSolveBenchmark(xs, us, problem)
print(f"  DDP.solve [ms]: {avrg_dur} ({min_dur}, {max_dur})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcBenchmark(xs, us, problem)
print(f"  ShootingProblem.calc [ms]: {avrg_dur} ({min_dur}, {max_dur})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcDiffBenchmark(xs, us, problem)
print(f"  ShootingProblem.calcDiff [ms]: {avrg_dur} ({min_dur}, {max_dur})")

print("Python:")
xs, us, problem = createProblem(LQRModelDerived)
avrg_dur, min_dur, max_dur = runDDPSolveBenchmark(xs, us, problem)
print(f"  DDP.solve [ms]: {avrg_dur} ({min_dur}, {max_dur})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcBenchmark(xs, us, problem)
print(f"  ShootingProblem.calc [ms]: {avrg_dur} ({min_dur}, {max_dur})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcDiffBenchmark(xs, us, problem)
print(f"  ShootingProblem.calcDiff [ms]: {avrg_dur} ({min_dur}, {max_dur})")
print("\033[0m")
