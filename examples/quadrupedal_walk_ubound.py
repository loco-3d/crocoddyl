from __future__ import print_function

import os
import sys
import time

import example_robot_data
import numpy as np

import crocoddyl
import pinocchio
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

# Loading the anymal model
anymal = example_robot_data.load('anymal')
robot_model = anymal.model
lims = robot_model.effortLimit
lims *= 0.5  # reduced artificially the torque limits
robot_model.effortLimit = lims

# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = 'LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT'
gait = SimpleQuadrupedalGaitProblem(robot_model, lfFoot, rfFoot, lhFoot, rhFoot)

# Defining the initial state of the robot
q0 = robot_model.referenceConfigurations['standing'].copy()
v0 = pinocchio.utils.zero(robot_model.nv)
x0 = np.concatenate([q0, v0])

# Defining the walking gait parameters
walking_gait = {'stepLength': 0.25, 'stepHeight': 0.25, 'timeStep': 1e-2, 'stepKnots': 25, 'supportKnots': 2}

# Setting up the control-limited DDP solver
boxddp = crocoddyl.SolverBoxDDP(
    gait.createWalkingProblem(x0, walking_gait['stepLength'], walking_gait['stepHeight'], walking_gait['timeStep'],
                              walking_gait['stepKnots'], walking_gait['supportKnots']))

# Add the callback functions
print('*** SOLVE ***')
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    boxddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    boxddp.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    boxddp.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
    ])
else:
    boxddp.setCallbacks([crocoddyl.CallbackVerbose()])

xs = [robot_model.defaultState] * (boxddp.problem.T + 1)
us = boxddp.problem.quasiStatic([anymal.model.defaultState] * boxddp.problem.T)

# Solve the DDP problem
boxddp.solve(xs, us, 100, False, 0.1)

# Plotting the entire motion
if WITHPLOT:
    # Plot control vs limits
    plotSolution(boxddp, bounds=True, figIndex=1, show=False)

    # Plot convergence
    log = boxddp.getCallbacks()[0]
    crocoddyl.plotConvergence(log.costs,
                              log.u_regs,
                              log.x_regs,
                              log.grads,
                              log.stops,
                              log.steps,
                              figIndex=3,
                              show=True)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(anymal, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    while True:
        display.displayFromSolver(boxddp)
        time.sleep(2.0)
