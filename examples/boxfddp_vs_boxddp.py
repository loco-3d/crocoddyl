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
lims = anymal.model.effortLimit
lims *= 0.4  # reduced artificially the torque limits
anymal.model.effortLimit = lims
lims = anymal.model.velocityLimit
lims *= 0.5
anymal.model.velocityLimit = lims

# Defining the initial state of the robot
q0 = anymal.model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(anymal.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
gait = SimpleQuadrupedalGaitProblem(anymal.model, lfFoot, rfFoot, lhFoot, rhFoot)

# Defining the walking gait parameters
jumping_gait = {
    "jumpHeight": 0.15,
    "jumpLength": [0.3, 0.0, 0.0],
    "timeStep": 1e-2,
    "groundKnots": 20,
    "flyingKnots": 20,
}

# Creating a both solvers
boxfddp = crocoddyl.SolverBoxFDDP(
    gait.createJumpingProblem(
        x0,
        jumping_gait["jumpHeight"],
        jumping_gait["jumpLength"],
        jumping_gait["timeStep"],
        jumping_gait["groundKnots"],
        jumping_gait["flyingKnots"],
    )
)
boxddp = crocoddyl.SolverBoxDDP(
    gait.createJumpingProblem(
        x0,
        jumping_gait["jumpHeight"],
        jumping_gait["jumpLength"],
        jumping_gait["timeStep"],
        jumping_gait["groundKnots"],
        jumping_gait["flyingKnots"],
    )
)

# Added the callback functions
if WITHPLOT:
    boxfddp.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
    boxddp.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
else:
    boxfddp.setCallbacks([crocoddyl.CallbackVerbose()])
    boxddp.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the both solvers
xs = [x0] * (boxfddp.problem.T + 1)
us = boxddp.problem.quasiStatic([x0] * boxddp.problem.T)

print("*** SOLVE with Box-FDDP ***")
boxfddp.th_stop = 1e-7
boxfddp.solve(xs, us, 50, False)

print("*** SOLVE with Box-DDP ***")
boxddp.th_stop = 1e-7
boxddp.solve(xs, us, 30, False)

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
        display.displayFromSolver(boxddp)
        time.sleep(1.0)
        display.displayFromSolver(boxfddp)
        time.sleep(1.0)

# Plotting the entire motion
if WITHPLOT:
    plotSolution(boxfddp, figIndex=1, figTitle="Box-FDDP", show=False)
    plotSolution(boxddp, figIndex=3, figTitle="Box-DDP", show=False)

    log = boxfddp.getCallbacks()[1]
    crocoddyl.plotConvergence(
        log.costs,
        log.pregs,
        log.dregs,
        log.grads,
        log.stops,
        log.steps,
        show=False,
        figTitle="Box-FDDP",
        figIndex=5,
    )
    log = boxddp.getCallbacks()[1]
    crocoddyl.plotConvergence(
        log.costs,
        log.pregs,
        log.dregs,
        log.grads,
        log.stops,
        log.steps,
        figTitle="Box-DDP",
        figIndex=7,
    )
