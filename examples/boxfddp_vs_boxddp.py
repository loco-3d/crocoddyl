import os
import sys

import crocoddyl
import numpy as np
import example_robot_data
from crocoddyl.utils.pendulum import CostModelDoublePendulum, ActuationModelDoublePendulum

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

crocoddyl.switchToNumpyMatrix()

# Loading the double pendulum model for testing both solvers
robot = example_robot_data.loadDoublePendulum()
robot_model = robot.model

state = crocoddyl.StateMultibody(robot_model)
actModel = ActuationModelDoublePendulum(state, actLink=0)

weights = np.array([1, 1, 1, 1] + [0.1] * 2)
runningCostModel = crocoddyl.CostModelSum(state, actModel.nu)
terminalCostModel = crocoddyl.CostModelSum(state, actModel.nu)
xRegCost = crocoddyl.CostModelState(state, crocoddyl.ActivationModelQuad(state.ndx), state.zero(), actModel.nu)
uRegCost = crocoddyl.CostModelControl(state, crocoddyl.ActivationModelQuad(1), actModel.nu)
xPendCost = CostModelDoublePendulum(state, crocoddyl.ActivationModelWeightedQuad(np.matrix(weights).T), actModel.nu)

dt = 1e-2

runningCostModel.addCost("uReg", uRegCost, 1e-4 / dt)
runningCostModel.addCost("xGoal", xPendCost, 1e-5 / dt)
terminalCostModel.addCost("xGoal", xPendCost, 1e4)

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actModel, terminalCostModel), dt)
runningModel.u_lb = np.matrix([-5])
runningModel.u_ub = np.matrix([5])

# Creating the shooting problem and the FDDP solver
T = 100
x0 = [3.14, 0, 0., 0.]
problem = crocoddyl.ShootingProblem(np.matrix(x0).T, [runningModel] * T, terminalModel)

# Creating both solvers: box-FDDP and box-DDP
boxfddp = crocoddyl.SolverBoxFDDP(problem)
boxddp = crocoddyl.SolverBoxDDP(problem)

cameraTF = [1.4, 0., 0.2, 0.5, 0.5, 0.5, 0.5]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF, False)
    boxfddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
    boxddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF, False)
    boxfddp.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
    boxddp.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    boxfddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
    boxddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])
else:
    boxfddp.setCallbacks([crocoddyl.CallbackVerbose()])
    boxddp.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the both solvers
boxfddp.th_stop = 1e-5
boxddp.th_stop = 1e-5
print("Solving the problem with Box-FDDP")
boxfddp.solve([], [], 500)
print()
print("Solving the problem with Box-DDP")
boxddp.solve([], [], 500)

# Plotting the entire motion
if WITHPLOT:
    log = boxfddp.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False, figTitle="Box-FDDP")
    crocoddyl.plotConvergence(log.costs,
                              log.u_regs,
                              log.x_regs,
                              log.grads,
                              log.stops,
                              log.steps,
                              show=False,
                              figIndex=2,
                              figTitle="Box-FDDP")
    log = boxddp.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=3, show=False, figTitle="Box-DDP")
    crocoddyl.plotConvergence(log.costs,
                              log.u_regs,
                              log.x_regs,
                              log.grads,
                              log.stops,
                              log.steps,
                              figIndex=4,
                              figTitle="Box-DDP")

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(robot, floor=False)
    display.displayFromSolver(boxfddp)
    display.displayFromSolver(boxddp)
