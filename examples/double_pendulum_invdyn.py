import os
import signal
import sys
import time

import crocoddyl
import example_robot_data
import numpy as np
from crocoddyl.utils.pendulum import (
    ActuationModelDoublePendulum,
    CostModelDoublePendulum,
)

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the double pendulum model
pendulum = example_robot_data.load("double_pendulum")
model = pendulum.model

state = crocoddyl.StateMultibody(model)
actuation = ActuationModelDoublePendulum(state, actLink=1)

nu = state.nv
runningCostModel = crocoddyl.CostModelSum(state, nu)
terminalCostModel = crocoddyl.CostModelSum(state, nu)

xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xActivation = crocoddyl.ActivationModelQuad(state.ndx)
uResidual = crocoddyl.ResidualModelJointEffort(
    state, actuation, np.zeros(actuation.nu), nu, False
)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
xPendCost = CostModelDoublePendulum(
    state, crocoddyl.ActivationModelWeightedQuad(np.array([1.0] * 4 + [0.1] * 2)), nu
)

dt = 1e-2

runningCostModel.addCost("uReg", uRegCost, 1e-4 / dt)
runningCostModel.addCost("xGoal", xPendCost, 1e-5 / dt)
terminalCostModel.addCost("xGoal", xPendCost, 100.0)

runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeInvDynamics(
        state, actuation, runningCostModel
    ),
    dt,
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeInvDynamics(
        state, actuation, terminalCostModel
    ),
    dt,
)

# Creating the shooting problem and the solver
T = 100
x0 = np.array([3.14, 0.0, 0.0, 0.0])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
solver = crocoddyl.SolverIntro(problem)

cameraTF = [1.4, 0.0, 0.2, 0.5, 0.5, 0.5, 0.5]
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        display = crocoddyl.GepettoDisplay(pendulum, 4, 4, cameraTF, floor=False)
        if WITHPLOT:
            solver.setCallbacks(
                [
                    crocoddyl.CallbackVerbose(),
                    crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackDisplay(display),
                ]
            )
        else:
            solver.setCallbacks(
                [crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)]
            )
    except Exception:
        display = crocoddyl.MeshcatDisplay(pendulum)
if WITHPLOT:
    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
solver.getCallbacks()[0].precision = 3
solver.getCallbacks()[0].level = crocoddyl.VerboseLevel._2

# Solving the problem with the solver
solver.solve()

# Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[1]
    crocoddyl.plotOCSolution(
        log.xs, [u[state.nv :] for u in log.us], figIndex=1, show=False
    )
    crocoddyl.plotConvergence(
        log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps, figIndex=2
    )

# Display the entire motion
if WITHDISPLAY:
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)
