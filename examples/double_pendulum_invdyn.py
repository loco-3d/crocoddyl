import os
import signal
import sys
import time

import example_robot_data
import numpy as np

import crocoddyl

if crocoddyl.WITH_ODYN:
    from odyn.utils import plotQPsparsity
from crocoddyl.utils.pendulum import ActuationModelDoublePendulum

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the double pendulum model
pendulum = example_robot_data.load("double_pendulum")
pendulum.model.upperPositionLimit[:] = np.pi
pendulum.model.lowerPositionLimit[:] = -np.pi
pendulum.model.velocityLimit[:] = 15
pendulum.model.effortLimit[:] = 5.0

# Creating the state and actuaction models
state = crocoddyl.StateMultibody(pendulum.model)
actuation = ActuationModelDoublePendulum(state, actLink=1)
nu, dt = state.nv, 1e-2

# Defining the residuals, costs, and constraints
target_state = state.zero()
goalResidual = crocoddyl.ResidualModelState(state, target_state, nu)
xResidual = crocoddyl.ResidualModelState(state, target_state, nu)
xActivation = crocoddyl.ActivationModelQuad(state.ndx)
uResidual = crocoddyl.ResidualModelJointEffort(
    state, actuation, np.zeros(actuation.nu), nu, False
)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
xGoalConstraint = crocoddyl.ConstraintModelResidual(state, goalResidual)

# Adding the costs and constraints
runningCosts = crocoddyl.CostModelSum(state, nu)
terminalCosts = crocoddyl.CostModelSum(state, nu)
terminalConstraints = crocoddyl.ConstraintModelManager(state, nu)
runningCosts.addCost("uReg", uRegCost, 1e-4 / dt)
runningCosts.addCost("xGoal", xRegCost, 1e-5 / dt)
terminalCosts.addCost("xGoal", xRegCost, 1e4)
terminalConstraints.addConstraint("xGoal", xGoalConstraint)

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeInvDynamics(state, actuation, runningCosts), dt
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeInvDynamics(
        state, actuation, terminalCosts, terminalConstraints
    ),
    dt,
)

# Creating the shooting problem and the OC solver
T = 100
x0 = np.array([3.14, 0.0, 0.0, 0.0])
problem_1 = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
problem_2 = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
solver = crocoddyl.SolverIntro(problem_1)
if crocoddyl.WITH_ODYN:
    solverSQP = crocoddyl.SolverOdynSQP(problem_2)
if WITHPLOT:
    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
    if crocoddyl.WITH_ODYN:
        solverSQP.setCallbacks(
            [crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()]
        )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    if crocoddyl.WITH_ODYN:
        solverSQP.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the OC solver
xs = [problem_1.x0] * (T + 1)
print("*** SOLVE (FeasShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
solver.solve(xs, [], 300)
print("*** SOLVE (MultiShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve(xs, [], 300)
Ts = int(solver.problem.T / 3)
print("*** SOLVE (HybridShoot: {Ts}) ***".format_map(locals()))
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
solver.solve(xs, [], 300)
if crocoddyl.WITH_ODYN:
    print("*** SOLVE (OdynSQP) ***")
    solverSQP.solve(xs, [], 300)

# Printing the terminal state
np.set_printoptions(precision=4, suppress=True)
print("Target state:", target_state)
print("Terminal state:", solver.xs[-1])

# Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[1]
    xs, us = (
        solver.xs,
        [d.differential.multibody.joint.tau for d in solver.problem.runningDatas],
    )
    crocoddyl.plotOCSolution(xs, us, figIndex=1, show=False)
    crocoddyl.plotConvergence(
        log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2
    )
    if crocoddyl.WITH_ODYN:
        logSQP = solverSQP.getCallbacks()[1]
        xs, us = (
            solverSQP.xs,
            [
                d.differential.multibody.joint.tau
                for d in solverSQP.problem.runningDatas
            ],
        )
        crocoddyl.plotOCSolution(xs, us, figIndex=3, show=False)
        crocoddyl.plotConvergence(
            logSQP.costs,
            logSQP.pregs,
            logSQP.dregs,
            logSQP.stops,
            logSQP.grads,
            logSQP.steps,
            figIndex=4,
        )
        plotQPsparsity(solverSQP.qp_model, figIndex=5, show=True)

# Display the entire motion
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        cameraTF = [1.4, 0.0, 0.2, 0.5, 0.5, 0.5, 0.5]
        display = crocoddyl.GepettoDisplay(pendulum, 4, 4, cameraTF, floor=False)
    except Exception:
        display = crocoddyl.MeshcatDisplay(pendulum)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)
