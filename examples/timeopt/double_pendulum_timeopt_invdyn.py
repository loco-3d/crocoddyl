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
model = pendulum.model

# Creating the state and actuation models
state = crocoddyl.StateMultibody(model)
actuation = ActuationModelDoublePendulum(state, actLink=1)
nu, dt = state.nv, 1e-2
np_total = 1
time_reg_weight = 1e1
runningTime = crocoddyl.IntegratorTime(dt, True)
terminalTime = runningTime

# Defining the residuals, costs, and constraints
target_state = state.zero()
xResidual = crocoddyl.ResidualModelState(state, target_state, nu)
xActivation = crocoddyl.ActivationModelQuad(state.ndx)
uResidual = crocoddyl.ResidualModelJointEffort(
    state, actuation, np.zeros(actuation.nu), nu, False
)
pResidual = crocoddyl.ResidualModelParameters(state, np.array([np.log(dt)]), nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
pRegCost = crocoddyl.CostModelResidual(state, pResidual)

goalResidual = crocoddyl.ResidualModelState(state, target_state, nu)
xGoalConstraint = crocoddyl.ConstraintModelResidual(state, goalResidual)

# Adding the costs and constraints
runningCosts = crocoddyl.CostModelSum(state, nu, np_total)
terminalCosts = crocoddyl.CostModelSum(state, nu, np_total)
terminalConstraints = crocoddyl.ConstraintModelManager(state, nu)
runningCosts.addCost("uReg", uRegCost, 1e-4 / dt)
runningCosts.addCost("xGoal", xRegCost, 1e-5 / dt)
runningCosts.addCost("pReg", pRegCost, time_reg_weight)
terminalConstraints.addConstraint("xGoal", xGoalConstraint)

# Creating the running and terminal models
runningDynamics = crocoddyl.DynamicsModelConstrainedInverse(
    state,
    actuation,
    crocoddyl.ImplicitConstraintModelMultiple(state, nu),
    np_total,
)
terminalDynamics = crocoddyl.DynamicsModelConstrainedInverse(
    state,
    actuation,
    crocoddyl.ImplicitConstraintModelMultiple(state, nu),
    np_total,
)
runningModel = crocoddyl.IntegratedActionModelEuler(
    runningDynamics,
    runningCosts,
    None,
    None,
    runningTime,
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    terminalDynamics,
    terminalCosts,
    terminalConstraints,
    None,
    terminalTime,
)

T = 100
x0 = np.array([0.0, -1.0, 0.0, -1.0, 0.0, 0.0])
p0 = np.array([np.log(dt)])

# Creating the parameter manager, shooting problem, and solver
params = crocoddyl.ParameterManager(state)
params.addParam("timeopt", crocoddyl.IntegratorTimeoptParams(state, runningTime))
problem = crocoddyl.ShootingProblem(
    x0,
    [runningModel] * T,
    terminalModel,
    crocoddyl.ParameterPhaseModel(params),
)
problem.update_p(p0, phase_idx=0)
solver = crocoddyl.SolverIntro(problem)
solver.th_minImprove = 1e-4
solver.th_stop = 1e-4
if WITHPLOT:
    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

xs = [x0] * (T + 1)
us = [np.zeros(model.nu) for model in problem.runningModels]
print("*** SOLVE (FeasShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
solver.solve(xs, us, [p0], 300, False)
print("*** SOLVE (MultiShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve(xs, us, [p0], 300, False)
Ts = int(solver.problem.T / 3)
print("*** SOLVE (HybridShoot: {Ts}) ***".format_map(locals()))
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
solver.solve(xs, us, [p0], 300, False)
print("step integration (guess): ", dt)
print("step integration (optimal): ", np.exp(solver.p[0][0]))

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
    except (RuntimeError, ImportError):
        display = crocoddyl.MeshcatDisplay(pendulum)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)
