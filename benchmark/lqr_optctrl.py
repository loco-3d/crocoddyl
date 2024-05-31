import sys
import time

import numpy as np

import crocoddyl

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
    solver = crocoddyl.SolverFDDP(problem)
    if CALLBACKS:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])
    duration = []
    for i in range(T):
        c_start = time.time()
        solver.solve(xs, us, MAXITER)
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


xs, us, problem = createProblem(crocoddyl.ActionModelLQR)
print("NQ:", problem.terminalModel.state.nq)
print("Number of nodes:", problem.T)
avrg_dur, min_dur, max_dur = runDDPSolveBenchmark(xs, us, problem)
print(f"  FDDP.solve [ms]: {avrg_dur:.4f} ({min_dur:.4f}-{max_dur:.4f})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcBenchmark(xs, us, problem)
print(f"  ShootingProblem.calc [ms]: {avrg_dur:.4f} ({min_dur:.4f}-{max_dur:.4f})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcDiffBenchmark(xs, us, problem)
print(f"  ShootingProblem.calcDiff [ms]: {avrg_dur:.4f} ({min_dur:.4f}-{max_dur:.4f})")
