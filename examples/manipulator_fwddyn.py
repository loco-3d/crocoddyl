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

# In this example test, we will solve the reaching-goal task with the Kinova arm.
# For that, we use the forward dynamics (with its analytical derivatives) developed
# inside crocoddyl; it is described inside DifferentialActionModelFreeFwdDynamics class.
# Finally, we use an Euler sympletic integration scheme.

# Loading the Kinova manipulator robot
kinova = example_robot_data.load("kinova")
kinova.model.lowerPositionLimit[3] = -np.pi
kinova.model.upperPositionLimit[3] = np.pi

# Creating the state and actuaction models
state = crocoddyl.StateMultibody(kinova.model)
actuation = crocoddyl.ActuationModelFull(state)
nv, nu, dt = state.nv, state.nv, 1e-2

q0 = state.pinocchio.referenceConfigurations["arm_up"]
x0 = np.concatenate([q0, pinocchio.utils.zero(nv)])

target_id = state.pinocchio.getFrameId("j2s6s200_end_effector")
target_pos = np.array([0.6, 0.2, 0.5])
target_rot = np.eye(3)
eePoseResidual = crocoddyl.ResidualModelFramePlacement(
    state,
    target_id,
    pinocchio.SE3(target_rot, target_pos),
    nu,
)
uResidual = crocoddyl.ResidualModelJointEffort(state, actuation, nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0.1] * state.nv + [10.0] * nv)
)
xResidual = crocoddyl.ResidualModelState(state, x0, nu)
accResidual = crocoddyl.ResidualModelJointAcceleration(state, nu)
eeTrackingCost = crocoddyl.CostModelResidual(state, eePoseResidual)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
accRegCost = crocoddyl.CostModelResidual(state, accResidual)
eePoseConstraint = crocoddyl.ConstraintModelResidual(state, eePoseResidual)

# Then let's added the running and terminal cost functions
runningCosts = crocoddyl.CostModelSum(state)
terminalCosts = crocoddyl.CostModelSum(state)
terminalConstraints = crocoddyl.ConstraintModelManager(state, nu)
runningCosts.addCost("xReg", xRegCost, 1e-1)
runningCosts.addCost("uReg", uRegCost, 1e-1)
runningCosts.addCost("accReg", accRegCost, 5e-1)
terminalConstraints.addConstraint("goalPose", eePoseConstraint)

# Creating the running and terminal models
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCosts), dt
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, terminalCosts, terminalConstraints
    ),
    0.0,
)

# Creating the shooting problem and the OC solver
T = 100
problem_1 = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
problem_2 = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)
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

# Solving it with the OC solver
xs = [x0] * (T + 1)
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

# Printing the terminal end-effector pose
Mterm = pinocchio.SE3ToXYZQUAT(
    solver.problem.terminalData.differential.multibody.pinocchio.oMf[target_id]
)
print("Target end-effector pose:")
print("   position:", target_pos)
print("   quaternion:", pinocchio.Quaternion(target_rot).coeffs())
print("Terminal end-effector pose:")
print("   position:", Mterm[:3])
print("   quaternion:", Mterm[3:7])

# Plotting the solution and the solver convergence
if WITHPLOT:
    log = solver.getCallbacks()[1]
    xs, us = solver.xs, solver.us
    crocoddyl.plotOCSolution(xs, us, figIndex=1, show=False)
    crocoddyl.plotConvergence(
        log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2
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

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    import meshcat.geometry as g

    display = crocoddyl.MeshcatDisplay(kinova)
    color = g.MeshLambertMaterial(
        color=display._rgbToHexColor([0.8156, 0.1569, 0.5686, 1.0]),
        reflectivity=0.8,
    )
    display.robot.viewer["target"].set_object(g.Sphere(0.015), color)
    display.robot.viewer["target"].set_transform(
        pinocchio.SE3(target_rot, target_pos).homogeneous
    )
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        if crocoddyl.WITH_ODYN:
            display.displayFromSolver(solverSQP)
        time.sleep(1.0)
