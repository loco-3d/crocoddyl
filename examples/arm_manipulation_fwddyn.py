import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# In this example test, we will solve the reaching-goal task with the Kinova arm.
# For that, we use the forward dynamics (with its analytical derivatives) developed
# inside crocoddyl; it is described inside DifferentialActionModelFreeFwdDynamics class.
# Finally, we use an Euler sympletic integration scheme.

# First, let's load create the state and actuation models
kinova = example_robot_data.load("kinova")
robot_model = kinova.model
state = crocoddyl.StateMultibody(robot_model)
actuation = crocoddyl.ActuationModelFull(state)
q0 = kinova.model.referenceConfigurations["arm_up"]
x0 = np.concatenate([q0, pinocchio.utils.zero(robot_model.nv)])

# Create a cost model per the running and terminal action model.
nu = state.nv
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

# Note that we need to include a cost model (i.e. set of cost functions) in
# order to fully define the action model for our optimal control problem.
# For this particular example, we formulate three running-cost functions:
# goal-tracking cost, state and control regularization; and one terminal-cost:
# goal cost. First, let's create the common cost functions.
framePlacementResidual = crocoddyl.ResidualModelFramePlacement(
    state,
    robot_model.getFrameId("j2s6s200_end_effector"),
    pinocchio.SE3(np.eye(3), np.array([0.6, 0.2, 0.5])),
    nu,
)
uResidual = crocoddyl.ResidualModelControl(state, nu)
xResidual = crocoddyl.ResidualModelState(state, x0, nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, framePlacementResidual)
xRegCost = crocoddyl.CostModelResidual(state, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1)
runningCostModel.addCost("xReg", xRegCost, 1e-1)
runningCostModel.addCost("uReg", uRegCost, 1e-1)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e3)

# Next, we need to create an action model for running and terminal knots. The
# forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFreeFwdDynamics.
dt = 1e-2
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, runningCostModel
    ),
    dt,
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, terminalCostModel
    ),
    0.0,
)

# For this optimal control problem, we define 100 knots (or running action
# models) plus a terminal knot
T = 100
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
solver = crocoddyl.SolverFDDP(problem)
if WITHPLOT:
    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving it with the solver algorithm
solver.solve()

print(
    "Finally reached = ",
    solver.problem.terminalData.differential.multibody.pinocchio.oMf[
        robot_model.getFrameId("j2s6s200_end_effector")
    ].translation.T,
)

# Plotting the solution and the solver convergence
if WITHPLOT:
    log = solver.getCallbacks()[1]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(
        log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2
    )

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    try:
        import gepetto

        cameraTF = [2.0, 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
        gepetto.corbaserver.Client()
        display = crocoddyl.GepettoDisplay(kinova, 4, 4, cameraTF, floor=False)
    except Exception:
        display = crocoddyl.MeshcatDisplay(kinova)

    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)
