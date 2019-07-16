import crocoddyl
import utils
import numpy as np
import time

NX = 37
NU = 12
N = 100  # number of nodes
T = int(5e3)  # number of trials
MAXITER = 1


def runBenchmark(model):
    x0 = np.matrix(np.zeros(NX)).T
    runningModels = [model(NX, NU)] * N
    terminalModel = model(NX, NU)
    xs = [x0] * (N + 1)
    us = [np.matrix(np.zeros(NU)).T] * N

    problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
    ddp = crocoddyl.SolverDDP(problem)

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


print('cpp-wrapped lqr:')
avrg_duration, min_duration, max_duration = runBenchmark(crocoddyl.ActionModelLQR)
print('  CPU time [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))

# print('Python-derived unicycle:')
# avrg_duration, min_duration, max_duration = runBenchmark(utils.UnicycleDerived)
# print('  CPU time [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
