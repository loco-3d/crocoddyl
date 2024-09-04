import os
import signal
import sys
import time

import example_robot_data
import numpy as np

import crocoddyl
from crocoddyl.utils.pendulum import ActuationModelDoublePendulum

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the double pendulum model
pendulum = example_robot_data.load("double_pendulum_continuous")

# Creating the state and actuaction models
state = crocoddyl.StateMultibody(pendulum.model)
actuation = ActuationModelDoublePendulum(state, actLink=1)
nu, dt = actuation.nu, 1e-2

# Defining the residuals, costs, and constraints
target_state = state.zero()
goalResidual = crocoddyl.ResidualModelState(state, target_state, nu)
xResidual = crocoddyl.ResidualModelState(state, target_state, nu)
uResidual = crocoddyl.ResidualModelJointEffort(state, actuation, nu)
xActivation = crocoddyl.ActivationModelQuad(state.ndx)
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

# Creating the running and terminal models
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCosts), dt
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, terminalCosts, terminalConstraints
    ),
    dt,
)

# Creating the shooting problem and the OC solver
T = 100
x0 = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 0.0])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
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

# Solving the problem with the FDDP solver
print("*** SOLVE (FeasShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
solver.solve([], [], 300)
print("*** SOLVE (MultiShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve([], [], 300)
Ts = int(solver.problem.T / 3)
print("*** SOLVE (HybridShoot: {Ts}) ***".format_map(locals()))
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
solver.solve([], [], 300)

# Printing the terminal state
np.set_printoptions(precision=4, suppress=True)
print("Target state:", target_state)
print("Terminal state:", solver.xs[-1])

# Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[1]
    crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(
        log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2
    )

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
