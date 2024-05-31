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

# Loading the anymal model
anymal = example_robot_data.load("anymal")
robot_model = anymal.model
lims = robot_model.effortLimit
lims *= 0.5  # reduced artificially the torque limits
robot_model.effortLimit = lims

# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
gait = SimpleQuadrupedalGaitProblem(robot_model, lfFoot, rfFoot, lhFoot, rhFoot)

# Defining the initial state of the robot
q0 = robot_model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(robot_model.nv)
x0 = np.concatenate([q0, v0])

# Defining the walking gait parameters
walking_gait = {
    "stepLength": 0.25,
    "stepHeight": 0.25,
    "timeStep": 1e-2,
    "stepKnots": 25,
    "supportKnots": 2,
}

# Setting up the control-limited DDP solver
solver = crocoddyl.SolverBoxDDP(
    gait.createWalkingProblem(
        x0,
        walking_gait["stepLength"],
        walking_gait["stepHeight"],
        walking_gait["timeStep"],
        walking_gait["stepKnots"],
        walking_gait["supportKnots"],
    )
)

# Add the callback functions
print("*** SOLVE ***")
if WITHPLOT:
    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solve the DDP problem
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
solver.solve(xs, us, 100, False, 0.1)

# Plotting the entire motion
if WITHPLOT:
    # Plot control vs limits
    plotSolution(solver, bounds=True, figIndex=1, show=False)

    # Plot convergence
    log = solver.getCallbacks()[1]
    crocoddyl.plotConvergence(
        log.costs,
        log.pregs,
        log.dregs,
        log.grads,
        log.stops,
        log.steps,
        figIndex=3,
        show=True,
    )

# Display the entire motion
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        cameraTF = [2.0, 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
        display = crocoddyl.GepettoDisplay(anymal, 4, 4, cameraTF)
    except Exception:
        display = crocoddyl.MeshcatDisplay(anymal)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)
