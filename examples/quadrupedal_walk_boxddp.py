from __future__ import print_function

import time

import example_robot_data
import numpy as np
import matplotlib.pyplot as plt

import crocoddyl
import pinocchio
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem
from crocoddyl import plotConvergence

robot_model = example_robot_data.loadHyQ().model
lfFoot, rfFoot, lhFoot, rhFoot = 'lf_foot', 'rf_foot', 'lh_foot', 'rh_foot'
gait = SimpleQuadrupedalGaitProblem(robot_model, lfFoot, rfFoot, lhFoot, rhFoot)
q0 = robot_model.referenceConfigurations['half_sitting'].copy()
v0 = pinocchio.utils.zero(robot_model.nv)
x0 = np.concatenate([q0, v0])

walking_gait = {'stepLength': 0.25, 'stepHeight': 0.25, 'timeStep': 1e-2, 'stepKnots': 25, 'supportKnots': 2}

boxddp = crocoddyl.SolverBoxDDP(
    gait.createWalkingProblem(x0, walking_gait['stepLength'], walking_gait['stepHeight'], walking_gait['timeStep'],
                              walking_gait['stepKnots'], walking_gait['supportKnots']))
boxddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

xs = [robot_model.defaultState] * len(boxddp.models())
us = [m.quasiStatic(d, robot_model.defaultState) for m, d in list(zip(boxddp.models(), boxddp.datas()))[:-1]]

boxddp_start = time.time()
boxddp.solve(xs, us, 1000, False, 0.1)
boxddp_end = time.time()
print("[Box-DDP] Solved in", boxddp_end - boxddp_start, "-", boxddp.iter, "iterations")

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
