import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data
from crocoddyl.utils.quadrotor import ActuationModelUAM

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

crocoddyl.switchToNumpyMatrix()

hector = example_robot_data.loadHector()
robot_model = hector.model

target_pos = np.array([1, 0, 1])
target_quat = pinocchio.Quaternion(1, 0, 0, 0)

state = crocoddyl.StateMultibody(robot_model)
distance_rotor_COG = 0.1525
cf = 6.6e-5
cm = 1e-6
u_lim = 5
l_lim = 0.1
actModel = ActuationModelUAM(state, '+', distance_rotor_COG, cm, cf, u_lim, l_lim)

runningCostModel = crocoddyl.CostModelSum(state, actModel.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actModel.nu)

# Needed objects to create the costs
Mref = crocoddyl.FramePlacement(robot_model.getFrameId("base_link"),
                                pinocchio.SE3(target_quat.matrix(),
                                              np.matrix(target_pos).T))
wBasePos, wBaseOri, wBaseVel, wBaseRate = [0.1], [1000], [1000], [10]
stateWeights = np.matrix([wBasePos * 3 + wBaseOri * 3 + wBaseVel * robot_model.nv]).T

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

# Creating the shooting problem and the FDDP solver
T = 33
problem = crocoddyl.ShootingProblem(np.vstack([hector.q0, pinocchio.utils.zero((state.nv))]), [runningModel] * T,
                                    terminalModel)
fddp = crocoddyl.SolverFDDP(problem)

fddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

cameraTF = [-0.03, 4.4, 2.3, -0.02, 0.56, 0.83, -0.03]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(hector, 4, 4, cameraTF)
    fddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(hector, 4, 4, cameraTF)
    fddp.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    fddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    fddp.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the FDDP solver
fddp.solve()

# Plotting the entire motion
if WITHPLOT:
    log = fddp.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs,
                              log.control_regs,
                              log.state_regs,
                              log.gm_stops,
                              log.th_stops,
                              log.steps,
                              figIndex=2)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(hector)
    hector.viewer.gui.addXYZaxis('world/wp', [1., 0., 0., 1.], .03, 0.5)
    hector.viewer.gui.applyConfiguration(
        'world/wp',
        target_pos.tolist() + [target_quat[0], target_quat[1], target_quat[2], target_quat[3]])

    display.displayFromSolver(fddp)
