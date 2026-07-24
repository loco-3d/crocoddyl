import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the B1 quadruped model
robot = example_robot_data.load("b1")

# Defining the initial state of the robot
q0 = robot.model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(robot.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = "FL_foot", "FR_foot", "RL_foot", "RR_foot"
n_phases = 12
gait = SimpleQuadrupedalGaitProblem(
    robot.model,
    lfFoot,
    rfFoot,
    lhFoot,
    rhFoot,
    termConstraint=True,
    timeopt=True,
    n_phases=n_phases,
    time_reg_weight=1e1,
)

# Setting up all tasks
GAITPHASES = {
    "stepLength": 0.25,
    "stepHeight": 0.15,
    "timeStep": 1e-2,
    "stepKnots": 25,
    "supportKnots": 2,
}

problem = gait.createWalkingProblem(
    x0,
    GAITPHASES["stepLength"],
    GAITPHASES["stepHeight"],
    GAITPHASES["timeStep"],
    GAITPHASES["stepKnots"],
    GAITPHASES["supportKnots"],
)

# Creating the solver
solver = crocoddyl.SolverFDDP(problem)
solver.th_minImprove = 1e-4
solver.th_stop = 1e-4
if WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the FDDP solver
dt = GAITPHASES["timeStep"]
p0 = np.array([np.log(dt)])
init_p = [p0.copy() for _ in range(n_phases)]
for i, p in enumerate(init_p):
    problem.update_p(p, phase_idx=i)
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)

print("*** SOLVE (FeasShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
solver.solve(xs, us, init_p, 100, False)
print("*** SOLVE (MultiShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve(xs, us, init_p, 100, False)
Ts = int(solver.problem.T / 3)
print("*** SOLVE (HybridShoot: {Ts}) ***".format_map(locals()))
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
solver.solve(xs, us, init_p, 100, False)

print("step integration (guess): ", dt)
print("step integration (optimal): ", np.exp(np.array([p[0] for p in solver.p])))

if WITHDISPLAY:
    display = crocoddyl.MeshcatDisplay(robot)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)

if WITHPLOT:
    plotSolution(solver, bounds=False, figIndex=1, show=False)
    log = solver.getCallbacks()[1]
    crocoddyl.plotConvergence(
        log.costs,
        log.pregs,
        log.dregs,
        log.grads,
        log.stops,
        log.steps,
        figIndex=3,
        show=False,
    )
