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

# Loading the Kinova manipulator robot
kinova = example_robot_data.load("kinova")
kinova.model.lowerPositionLimit[3] = -np.pi
kinova.model.upperPositionLimit[3] = np.pi

# Creating the state and actuation models
state = crocoddyl.StateMultibody(kinova.model)
actuation = crocoddyl.ActuationModelMultibody(state)
nv, nu, dt = state.nv, state.nv, 1e-2
np_total = 1
time_reg_weight = 3e1
runningTime = crocoddyl.IntegratorTime(dt, True)
terminalTime = runningTime

q0 = state.pinocchio.referenceConfigurations["arm_up"]
x0 = np.concatenate([q0, pinocchio.utils.zero(nv)])

# Defining the residuals, costs, and constraints
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
pResidual = crocoddyl.ResidualModelParameters(state, np.array([np.log(dt)]), nu)
eeTrackingCost = crocoddyl.CostModelResidual(state, eePoseResidual)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
accRegCost = crocoddyl.CostModelResidual(state, accResidual)
pRegCost = crocoddyl.CostModelResidual(state, pResidual)
eePoseConstraint = crocoddyl.ConstraintModelResidual(state, eePoseResidual)

# Adding the costs and constraints
runningCosts = crocoddyl.CostModelSum(state, nu, np_total)
terminalCosts = crocoddyl.CostModelSum(state, nu, np_total)
terminalConstraints = crocoddyl.ConstraintModelManager(state, nu)
runningCosts.addCost("xReg", xRegCost, 1e-1)
runningCosts.addCost("uReg", uRegCost, 1e-1)
runningCosts.addCost("accReg", accRegCost, 5e-1)
runningCosts.addCost("pReg", pRegCost, time_reg_weight)
terminalConstraints.addConstraint("goalPose", eePoseConstraint)

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

# Creating the parameter manager, shooting problem, and solver
T = 100
p0 = np.array([np.log(dt)])
params = crocoddyl.ParameterManager(state)
params.addParam("timeopt", crocoddyl.IntegratorTimeoptParams(state, runningTime))
problem = crocoddyl.ParametrizedShootingProblem(
    x0, [runningModel] * T, terminalModel, params
)
problem.update_p(p0, phase_idx=0)
solver = crocoddyl.SolverIntro(problem)
solver.th_minImprove = 1e-4
solver.th_stop = 1e-4
if WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the INTRO solver
xs = [x0] * (T + 1)
print("*** SOLVE (FeasShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
solver.solve(xs, [], [p0], 200)
print("*** SOLVE (MultiShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve(xs, [], [p0], 200)
Ts = int(solver.problem.T / 3)
print("*** SOLVE (HybridShoot: {Ts}) ***".format_map(locals()))
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
solver.solve(xs, [], [p0], 200)

print("step integration (guess): ", dt)
print("step integration (optimal): ", np.exp(solver.p[0][0]))

# Printing the terminal end-effector pose
pinocchio_data = state.pinocchio.createData()
pinocchio.forwardKinematics(state.pinocchio, pinocchio_data, solver.xs[-1][: state.nq])
pinocchio.updateFramePlacements(state.pinocchio, pinocchio_data)
Mterm = pinocchio.SE3ToXYZQUAT(pinocchio_data.oMf[target_id])
print("Target end-effector pose:")
print("   position:", target_pos)
print("   quaternion:", pinocchio.Quaternion(target_rot).coeffs())
print("Terminal end-effector pose:")
print("   position:", Mterm[:3])
print("   quaternion:", Mterm[3:7])

if WITHPLOT:
    log = solver.getCallbacks()[1]
    crocoddyl.plotOCSolution(solver.xs, solver.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(
        log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2
    )

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
        time.sleep(1.0)
