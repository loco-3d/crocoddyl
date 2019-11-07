from __future__ import print_function

import os
import sys
import time

import example_robot_data
import numpy as np
import matplotlib.pyplot as plt

import crocoddyl
import pinocchio
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

crocoddyl.switchToNumpyMatrix()
# Loading the anymal model
anymal = example_robot_data.loadANYmal()
robot_model = anymal.model

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
    boxddp.setCallbacks(
        [crocoddyl.CallbackLogger(),
         crocoddyl.CallbackVerbose(),
         crocoddyl.CallbackDisplay(anymal, 4, 4, cameraTF)])
elif WITHDISPLAY:
    boxddp.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(anymal, 4, 4, cameraTF)])
elif WITHPLOT:
    boxddp.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
    ])
else:
    boxddp.setCallbacks([crocoddyl.CallbackVerbose()])

xs = [robot_model.defaultState] * len(boxddp.models())
us = [m.quasiStatic(d, robot_model.defaultState) for m, d in list(zip(boxddp.models(), boxddp.datas()))[:-1]]

# Solve the DDP problem
boxddp_start = time.time()
boxddp.solve(xs, us, 1000, False, 0.1)
boxddp_end = time.time()
print("[Box-DDP] Solved in", boxddp_end - boxddp_start, "-", boxddp.iter, "iterations")

# Plotting the entire motion
if WITHPLOT:
    # Plot control vs limits
    fig = plt.figure(1)
    plt.title('Control ($u$)')
    plt.plot(np.asarray(boxddp.us)[:, :, 0])
    plt.xlim(0, len(boxddp.us) - 1)
    plt.hlines(boxddp.models()[0].u_lb, 0, len(boxddp.us) - 1, 'r')
    plt.hlines(boxddp.models()[0].u_ub, 0, len(boxddp.us) - 1, 'r')
    plt.tight_layout()

    # Plot convergence
    log = boxddp.getCallbacks()[0]
    crocoddyl.plotConvergence(log.costs,
                              log.control_regs,
                              log.state_regs,
                              log.gm_stops,
                              log.th_stops,
                              log.steps,
                              figIndex=2)

    plt.show()

# Display the entire motion
if WITHDISPLAY:
    while True:
        crocoddyl.displayTrajectory(anymal, boxddp.xs, boxddp.models()[0].dt)
        time.sleep(2.0)
