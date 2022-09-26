from __future__ import print_function

import os
import sys

import crocoddyl
import pinocchio
import numpy as np
import example_robot_data

WITHDISPLAY = 'display' in sys.argv or 'CROCODDYL_DISPLAY' in os.environ
WITHPLOT = 'plot' in sys.argv or 'CROCODDYL_PLOT' in os.environ

# In this example test, we will solve the reaching-goal task with the Talos arm.
# For that, we use the forward dynamics (with its analytical derivatives)
# developed inside crocoddyl; it describes inside DifferentialActionModelFullyActuated class.
# Finally, we use an Euler sympletic integration scheme.

# First, let's load create the state and actuation models
talos_arm = example_robot_data.load('talos_arm')
robot_model = talos_arm.model
state = crocoddyl.StateMultibody(robot_model)
actuation = crocoddyl.ActuationModelFull(state)

# Create a cost model per the running and terminal action model.
nu = state.nv + actuation.nu
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

# Note that we need to include a cost model (i.e. set of cost functions) in
# order to fully define the action model for our optimal control problem.
# For this particular example, we formulate three running-cost functions:
# goal-tracking cost, state and control regularization; and one terminal-cost:
# goal cost. First, let's create the common cost functions.
framePlacementResidual = crocoddyl.ResidualModelFramePlacement(state, robot_model.getFrameId("gripper_left_joint"),
                                                               pinocchio.SE3(np.eye(3), np.array([.0, .0, .4])), nu)
uResidual = crocoddyl.ResidualModelControl(state, nu)
xResidual = crocoddyl.ResidualModelControl(state, nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
uRegCost = crocoddyl.CostModelResidual(
    state, crocoddyl.ActivationModelWeightedQuad(np.array([1.] * state.nv + [0.] * actuation.nu)**2), uResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1)
runningCostModel.addCost("xReg", xRegCost, 1e-2)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e3)

# Next, we need to create an action model for running and terminal knots. The
# forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFullyActuated.

dt = 1e-3
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeInvDynamics(state, actuation, runningCostModel), dt)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeInvDynamics(state, actuation, terminalCostModel), 0.)

# For this optimal control problem, we define 250 knots (or running action
# models) plus a terminal knot
T = 250
q0 = np.array([0.173046, 1., -0.52366, 0., 0., 0.1, -0.005])
x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the solver for this OC problem, defining a logger
solver = crocoddyl.SolverIntro(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
if WITHDISPLAY and WITHPLOT:
    display = crocoddyl.GepettoDisplay(talos_arm, 4, 4, cameraTF)
    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(talos_arm, 4, 4, cameraTF)
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)])
elif WITHPLOT:
    solver.setCallbacks([
        crocoddyl.CallbackLogger(),
        crocoddyl.CallbackVerbose(),
    ])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving it with the solver algorithm
solver.solve()

print(
    'Finally reached = ', solver.problem.terminalData.differential.multibody.pinocchio.oMf[robot_model.getFrameId(
        "gripper_left_joint")].translation.T)

# Plotting the solution and the DDP convergence
if WITHPLOT:
    log = solver.getCallbacks()[0]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    display = crocoddyl.GepettoDisplay(talos_arm, 4, 4, cameraTF)
    display.displayFromSolver(solver)
