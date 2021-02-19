import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

hector = example_robot_data.load('hector')
robot_model = hector.model

target_pos = np.array([1, 0, 1])
target_quat = pinocchio.Quaternion(1, 0, 0, 0)

state = crocoddyl.StateMultibody(robot_model)
d_cog = 0.1525
cf = 6.6e-5
cm = 1e-6
u_lim = 5
l_lim = 0.1
tau_f = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.0, d_cog, 0.0, -d_cog],
                  [-d_cog, 0.0, d_cog, 0.0], [-cm / cf, cm / cf, -cm / cf, cm / cf]])

actModel = crocoddyl.ActuationModelMultiCopterBase(state, tau_f)

runningCostModel = crocoddyl.CostModelSum(state, actModel.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actModel.nu)

# Needed objects to create the costs
Mref = crocoddyl.FramePlacement(robot_model.getFrameId("base_link"), pinocchio.SE3(target_quat.matrix(), target_pos))
wBasePos, wBaseOri, wBaseVel = 0.1, 1000, 1000
stateWeights = np.array([wBasePos] * 3 + [wBaseOri] * 3 + [wBaseVel] * robot_model.nv)

# Costs
goalTrackingCost = crocoddyl.CostModelFramePlacement(state, Mref, actModel.nu)
xRegCost = crocoddyl.CostModelState(state, crocoddyl.ActivationModelWeightedQuad(stateWeights), state.zero(),
                                    actModel.nu)
uRegCost = crocoddyl.CostModelControl(state, actModel.nu)
runningCostModel.addCost("xReg", xRegCost, 1e-6)
runningCostModel.addCost("uReg", uRegCost, 1e-6)
runningCostModel.addCost("trackPose", goalTrackingCost, 1e-2)
terminalCostModel.addCost("goalPose", goalTrackingCost, 100)

dt = 3e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, terminalCostModel), dt)
runningModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
runningModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])

# Creating the shooting problem and the boxddp solver
T = 33
problem = crocoddyl.ShootingProblem(np.concatenate([hector.q0, np.zeros(state.nv)]), [runningModel] * T, terminalModel)
boxddp = crocoddyl.SolverBoxDDP(problem)
boxddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

cameraTF = [-0.03, 4.4, 2.3, -0.02, 0.56, 0.83, -0.03]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(hector, 4, 4, cameraTF)
    boxddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(hector, 4, 4, cameraTF)
    boxddp.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    boxddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    boxddp.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the boxddp solver
boxddp.solve([], [], 200)

# Plotting the entire motion
if WITHPLOT:
    log = boxddp.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(hector)
    hector.viewer.gui.addXYZaxis('world/wp', [1., 0., 0., 1.], .03, 0.5)
    hector.viewer.gui.applyConfiguration(
        'world/wp',
        target_pos.tolist() + [target_quat[0], target_quat[1], target_quat[2], target_quat[3]])

    display.displayFromSolver(boxddp)
