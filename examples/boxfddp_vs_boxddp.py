import os
import sys

import numpy as np

import crocoddyl
import example_robot_data
import pinocchio
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

# Loading the anymal model
anymal = example_robot_data.load('anymal')
lims = anymal.model.effortLimit
lims *= 0.4  # reduced artificially the torque limits
anymal.model.effortLimit = lims
lims = anymal.model.velocityLimit
lims *= 0.5
anymal.model.velocityLimit = lims

# Defining the initial state of the robot
q0 = anymal.model.referenceConfigurations['standing'].copy()
v0 = pinocchio.utils.zero(anymal.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = 'LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT'
gait = SimpleQuadrupedalGaitProblem(anymal.model, lfFoot, rfFoot, lhFoot, rhFoot)

# Defining the walking gait parameters
jumping_gait = {
    'jumpHeight': 0.15,
    'jumpLength': [0.3, 0., 0.],
    'timeStep': 1e-2,
    'groundKnots': 20,
    'flyingKnots': 20
}

cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]

# Creating a both solvers
boxfddp = crocoddyl.SolverBoxFDDP(
    gait.createJumpingProblem(x0, jumping_gait['jumpHeight'], jumping_gait['jumpLength'], jumping_gait['timeStep'],
                              jumping_gait['groundKnots'], jumping_gait['flyingKnots']))
boxddp = crocoddyl.SolverBoxDDP(
    gait.createJumpingProblem(x0, jumping_gait['jumpHeight'], jumping_gait['jumpLength'], jumping_gait['timeStep'],
                              jumping_gait['groundKnots'], jumping_gait['flyingKnots']))

# Added the callback functions
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    boxfddp.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger(), crocoddyl.CallbackDisplay(display)])
    boxddp.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    boxfddp.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
    boxddp.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    boxfddp.setCallbacks([
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackLogger(),
    ])
    boxddp.setCallbacks([
        crocoddyl.CallbackVerbose(),
        crocoddyl.CallbackLogger(),
    ])
else:
    boxfddp.setCallbacks([crocoddyl.CallbackVerbose()])
    boxddp.setCallbacks([crocoddyl.CallbackVerbose()])
boxfddp.getCallbacks()[0].precision = 3
boxfddp.getCallbacks()[0].level = crocoddyl.VerboseLevel._2
boxddp.getCallbacks()[0].precision = 3
boxddp.getCallbacks()[0].level = crocoddyl.VerboseLevel._2

# Solving the problem with the both solvers
xs = [x0] * (boxfddp.problem.T + 1)
us = boxddp.problem.quasiStatic([x0] * boxddp.problem.T)

print('*** SOLVE with Box-FDDP ***')
boxfddp.th_stop = 1e-7
boxfddp.solve(xs, us, 50, False)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(anymal, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    display.displayFromSolver(boxfddp)

print('*** SOLVE with Box-DDP ***')
boxddp.th_stop = 1e-7
boxddp.solve(xs, us, 30, False)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(anymal, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    display.displayFromSolver(boxddp)

# Plotting the entire motion
if WITHPLOT:
    plotSolution(boxfddp, figIndex=1, figTitle="Box-FDDP", show=False)
    plotSolution(boxddp, figIndex=3, figTitle="Box-DDP", show=False)

    log = boxfddp.getCallbacks()[1]
    crocoddyl.plotConvergence(log.costs,
                              log.u_regs,
                              log.x_regs,
                              log.grads,
                              log.stops,
                              log.steps,
                              show=False,
                              figTitle="Box-FDDP",
                              figIndex=5)
    log = boxddp.getCallbacks()[1]
    crocoddyl.plotConvergence(log.costs,
                              log.u_regs,
                              log.x_regs,
                              log.grads,
                              log.stops,
                              log.steps,
                              figTitle="Box-DDP",
                              figIndex=7)
