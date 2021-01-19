import os
import sys

import crocoddyl
import numpy as np
import example_robot_data
from crocoddyl.utils.pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

# Loading the double pendulum model
robot = example_robot_data.load('double_pendulum')
robot_model = robot.model

state = crocoddyl.StateMultibody(robot_model)
actModel = ActuationModelDoublePendulum(state, actLink=1)

weights = np.array([1, 1, 1, 1] + [0.1] * 2)
runningCostModel = crocoddyl.CostModelSum(state, actModel.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actModel.nu)
xRegCost = crocoddyl.CostModelState(state, crocoddyl.ActivationModelQuad(state.ndx), state.zero(), actModel.nu)
uRegCost = crocoddyl.CostModelControl(state, crocoddyl.ActivationModelQuad(1), actModel.nu)
xPendCost = CostModelDoublePendulum(state, crocoddyl.ActivationModelWeightedQuad(weights), actModel.nu)

dt = 1e-2

runningCostModel.addCost("uReg", uRegCost, 1e-4 / dt)
runningCostModel.addCost("xGoal", xPendCost, 1e-5 / dt)
terminalCostModel.addCost("xGoal", xPendCost, 1e4)

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, terminalCostModel), dt)

# Creating the shooting problem and the FDDP solver
T = 100
x0 = np.array([3.14, 0, 0., 0.])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
fddp = crocoddyl.SolverFDDP(problem)

cameraTF = [1.4, 0., 0.2, 0.5, 0.5, 0.5, 0.5]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF, False)
    fddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF, False)
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
    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(robot, floor=False)
    display.displayFromSolver(fddp)
