import os
import sys

import numpy as np

import crocoddyl
import example_robot_data
import pinocchio
from crocoddyl.utils.biped import SimpleBipedGaitProblem, plotSolution

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

# Creating the lower-body part of Talos
talos_legs = example_robot_data.loadTalosLegs()

# Defining the initial state of the robot
q0 = talos_legs.model.referenceConfigurations['half_sitting'].copy()
v0 = pinocchio.utils.zero(talos_legs.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
rightFoot = 'right_sole_link'
leftFoot = 'left_sole_link'
gait = SimpleBipedGaitProblem(talos_legs.model, rightFoot, leftFoot)

# Setting up all tasks
GAITPHASES = \
    [{'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.0375, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.0375, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.0375, 'stepKnots': 25, 'supportKnots': 1}},
     {'walking': {'stepLength': 0.6, 'stepHeight': 0.1,
                  'timeStep': 0.0375, 'stepKnots': 25, 'supportKnots': 1}}]
cameraTF = [3., 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]

ddp = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == 'walking':
            # Creating a walking problem
            ddp[i] = crocoddyl.SolverDDP(
                gait.createWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                          value['stepKnots'], value['supportKnots']))

    # Added the callback functions
    print('*** SOLVE ' + key + ' ***')
    if WITHDISPLAY:
        ddp[i].setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(talos_legs, 4, 4, cameraTF)])
    else:
        ddp[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the DDP solver
    ddp[i].th_stop = 1e-9
    xs = [talos_legs.model.defaultState] * len(ddp[i].models())
    us = [m.quasicStatic(d, talos_legs.model.defaultState) for m, d in list(zip(ddp[i].models(), ddp[i].datas()))[:-1]]
    ddp[i].solve(xs, us, 1000, False, 0.1)

    # Defining the final state as initial one for the next phase
    x0 = ddp[i].xs[-1]

# Display the entire motion
if WITHDISPLAY:
    for i, phase in enumerate(GAITPHASES):
        crocoddyl.displayTrajectory(talos_legs, ddp[i].xs, ddp[i].models()[0].dt)

# Plotting the entire motion
if WITHPLOT:
    xs = []
    us = []
    for i, phase in enumerate(GAITPHASES):
        xs.extend(ddp[i].xs)
        us.extend(ddp[i].us)
    plotSolution(talos_legs.model, xs, us)
