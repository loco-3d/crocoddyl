import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl

if crocoddyl.WITH_ODYN:
    from odyn.utils import plotQPsparsity

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the Hector quadrotor robot
hector = example_robot_data.load("hector")

# Creating the state and actuaction models
state = crocoddyl.StateMultibody(hector.model)
d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
ps = [
    crocoddyl.Thruster(
        pinocchio.SE3(np.eye(3), np.array([d_cog, 0, 0])),
        cm / cf,
        crocoddyl.ThrusterType.CCW,
        0,
        10,
    ),
    crocoddyl.Thruster(
        pinocchio.SE3(np.eye(3), np.array([0, d_cog, 0])),
        cm / cf,
        crocoddyl.ThrusterType.CW,
        0,
        10,
    ),
    crocoddyl.Thruster(
        pinocchio.SE3(np.eye(3), np.array([-d_cog, 0, 0])),
        cm / cf,
        crocoddyl.ThrusterType.CCW,
        0,
        10,
    ),
    crocoddyl.Thruster(
        pinocchio.SE3(np.eye(3), np.array([0, -d_cog, 0])),
        cm / cf,
        crocoddyl.ThrusterType.CW,
        0,
        10,
    ),
]
actuation = crocoddyl.ActuationModelFloatingBaseThrusters(state, ps)
nv, nu, dt = state.nv, actuation.nu, 3e-2

# Defining the residuals, costs, and constraints
target_pos = np.array([1.0, 0.0, 1.0])
target_quat = pinocchio.Quaternion(1.0, 0.0, 0.0, 0.0)
goalPoseResidual = crocoddyl.ResidualModelFramePlacement(
    state,
    state.pinocchio.getFrameId("base_link"),
    pinocchio.SE3(target_quat.matrix(), target_pos),
    nu,
)
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0.1] * 3 + [1000.0] * 3 + [1000.0] * nv)
)
uResidual = crocoddyl.ResidualModelJointEffort(state, actuation, nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, goalPoseResidual)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
eePoseConstraint = crocoddyl.ConstraintModelResidual(state, goalPoseResidual)

# Adding the costs and constraints
runningCosts = crocoddyl.CostModelSum(state, nu)
terminalCosts = crocoddyl.CostModelSum(state, nu)
terminalConstraints = crocoddyl.ConstraintModelManager(state, nu)
runningCosts.addCost("trackPose", goalTrackingCost, 1e-2)
runningCosts.addCost("xReg", xRegCost, 1e-6)
runningCosts.addCost("uReg", uRegCost, 1e-6)
terminalConstraints.addConstraint("goalPose", eePoseConstraint)

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
T = 33
problem_1 = crocoddyl.ShootingProblem(
    np.concatenate([hector.q0, np.zeros(state.nv)]), [runningModel] * T, terminalModel
)
problem_2 = crocoddyl.ShootingProblem(
    np.concatenate([hector.q0, np.zeros(state.nv)]), [runningModel] * T, terminalModel
)
solver = crocoddyl.SolverFDDP(problem_1)
if crocoddyl.WITH_ODYN:
    solverSQP = crocoddyl.SolverOdynSQP(problem_2)
if WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
    if crocoddyl.WITH_ODYN:
        solverSQP.setCallbacks(
            [crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()]
        )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    if crocoddyl.WITH_ODYN:
        solverSQP.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the solver
print("*** SOLVE (FeasShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
solver.solve()
print("*** SOLVE (MultiShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve()
Ts = int(solver.problem.T / 3)
print("*** SOLVE (HybridShoot: {Ts}) ***".format_map(locals()))
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
solver.solve()
if crocoddyl.WITH_ODYN:
    print("*** SOLVE (OdynSQP) ***")
    solverSQP.solve()

# Printing the terminal pose
np.set_printoptions(precision=4, suppress=True)
print("Target pose:")
print("   position:", target_pos)
print("   quaternion:", target_quat.coeffs())
print("Terminal pose:")
print("   position:", solver.xs[-1][:3])
print("   quaternion:", solver.xs[-1][3:7])

# Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[1]
    crocoddyl.plotOCSolution(solver.xs, solver.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(
        log.costs, log.pregs, log.dregs, log.stops, log.grads, log.steps, figIndex=2
    )
    if crocoddyl.WITH_ODYN:
        logSQP = solverSQP.getCallbacks()[1]
        crocoddyl.plotOCSolution(solverSQP.xs, solverSQP.us, figIndex=3, show=False)
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
    display = crocoddyl.MeshcatDisplay(hector)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        if crocoddyl.WITH_ODYN:
            display.displayFromSolver(solverSQP)
        time.sleep(1.0)
