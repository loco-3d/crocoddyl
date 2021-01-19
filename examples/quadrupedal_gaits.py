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

# Defining the initial state of the robot
q0 = anymal.model.referenceConfigurations['standing'].copy()
v0 = pinocchio.utils.zero(anymal.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
lfFoot, rfFoot, lhFoot, rhFoot = 'LF_FOOT', 'RF_FOOT', 'LH_FOOT', 'RH_FOOT'
gait = SimpleQuadrupedalGaitProblem(anymal.model, lfFoot, rfFoot, lhFoot, rhFoot)

# Setting up all tasks
GAITPHASES = [{
    'walking': {
        'stepLength': 0.25,
        'stepHeight': 0.15,
        'timeStep': 1e-2,
        'stepKnots': 25,
        'supportKnots': 2
    }
}, {
    'trotting': {
        'stepLength': 0.15,
        'stepHeight': 0.1,
        'timeStep': 1e-2,
        'stepKnots': 25,
        'supportKnots': 2
    }
}, {
    'pacing': {
        'stepLength': 0.15,
        'stepHeight': 0.1,
        'timeStep': 1e-2,
        'stepKnots': 25,
        'supportKnots': 5
    }
}, {
    'bounding': {
        'stepLength': 0.15,
        'stepHeight': 0.1,
        'timeStep': 1e-2,
        'stepKnots': 25,
        'supportKnots': 5
    }
}, {
    'jumping': {
        'jumpHeight': 0.15,
        'jumpLength': [0.0, 0.3, 0.],
        'timeStep': 1e-2,
        'groundKnots': 10,
        'flyingKnots': 20
    }
}]
cameraTF = [2., 2.68, 0.84, 0.2, 0.62, 0.72, 0.22]

ddp = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == 'walking':
            # Creating a walking problem
            ddp[i] = crocoddyl.SolverFDDP(
                gait.createWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                          value['stepKnots'], value['supportKnots']))
        elif key == 'trotting':
            # Creating a trotting problem
            ddp[i] = crocoddyl.SolverFDDP(
                gait.createTrottingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                           value['stepKnots'], value['supportKnots']))
        elif key == 'pacing':
            # Creating a pacing problem
            ddp[i] = crocoddyl.SolverFDDP(
                gait.createPacingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                         value['stepKnots'], value['supportKnots']))
        elif key == 'bounding':
            # Creating a bounding problem
            ddp[i] = crocoddyl.SolverFDDP(
                gait.createBoundingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                           value['stepKnots'], value['supportKnots']))
        elif key == 'jumping':
            # Creating a jumping problem
            ddp[i] = crocoddyl.SolverFDDP(
                gait.createJumpingProblem(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                                          value['groundKnots'], value['flyingKnots']))

    # Added the callback functions
    print('*** SOLVE ' + key + ' ***')
    if WITHDISPLAY and WITHPLOT:
        display = crocoddyl.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
        ddp[i].setCallbacks(
            [crocoddyl.CallbackLogger(),
             crocoddyl.CallbackVerbose(),
             crocoddyl.CallbackDisplay(display)])
    elif WITHDISPLAY:
        display = crocoddyl.GepettoDisplay(anymal, 4, 4, cameraTF, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
        ddp[i].setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
    elif WITHPLOT:
        ddp[i].setCallbacks([
            crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose(),
        ])
    else:
        ddp[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the DDP solver
    xs = [anymal.model.defaultState] * (ddp[i].problem.T + 1)
    us = ddp[i].problem.quasiStatic([anymal.model.defaultState] * ddp[i].problem.T)
    ddp[i].solve(xs, us, 100, False, 0.1)

    # Defining the final state as initial one for the next phase
    x0 = ddp[i].xs[-1]

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(anymal, frameNames=[lfFoot, rfFoot, lhFoot, rhFoot])
    for i, phase in enumerate(GAITPHASES):
        display.displayFromSolver(ddp[i])

# Plotting the entire motion
if WITHPLOT:
    log = ddp[0].getCallbacks()[0]
    plotSolution(ddp, figIndex=1, show=False)

    for i, phase in enumerate(GAITPHASES):
        title = list(phase.keys())[0] + " (phase " + str(i) + ")"
        log = ddp[i].getCallbacks()[0]
        crocoddyl.plotConvergence(log.costs,
                                  log.u_regs,
                                  log.x_regs,
                                  log.grads,
                                  log.stops,
                                  log.steps,
                                  figTitle=title,
                                  figIndex=i + 3,
                                  show=True if i == len(GAITPHASES) - 1 else False)
