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

# Loading the Hector quadrotor robot
hector = example_robot_data.load("hector")

# Creating the state and actuation models
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
joint_dynamics = [crocoddyl.JointDynamicsModelThruster(ps)]
root_joint_id = state.pinocchio.getJointId("root_joint")
for jid in range(1, state.pinocchio.njoints):
    if jid == root_joint_id:
        continue
    joint = state.pinocchio.joints[jid]
    joint_dynamics.append(crocoddyl.JointDynamicsModelIdentity(jid, joint.nq, joint.nv))
actuation = crocoddyl.ActuationModelMultibody(state, joint_dynamics)
nv, nu, dt = state.nv, state.nv, 3e-2
np_total = 1
time_reg_weight = 3e-1
runningTime = crocoddyl.IntegratorTime(dt, True)
terminalTime = runningTime

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
pResidual = crocoddyl.ResidualModelParameters(state, np.array([np.log(dt)]), nu)
goalTrackingCost = crocoddyl.CostModelResidual(state, goalPoseResidual)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
pRegCost = crocoddyl.CostModelResidual(state, pResidual)
eePoseConstraint = crocoddyl.ConstraintModelResidual(state, goalPoseResidual)

# Adding the costs and constraints
runningCosts = crocoddyl.CostModelSum(state, nu, np_total)
terminalCosts = crocoddyl.CostModelSum(state, nu, np_total)
terminalConstraints = crocoddyl.ConstraintModelManager(state, nu)
runningCosts.addCost("trackPose", goalTrackingCost, 1e-2)
runningCosts.addCost("xReg", xRegCost, 1e-6)
runningCosts.addCost("uReg", uRegCost, 1e-6)
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
T = 33
x0 = np.concatenate([hector.q0, np.zeros(state.nv)])
p0 = np.array([np.log(dt)])
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
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the INTRO solver
print("*** SOLVE (FeasShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
solver.solve([], [], [p0], 100)
print("*** SOLVE (MultiShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve([], [], [p0], 100)
Ts = int(solver.problem.T / 3)
print("*** SOLVE (HybridShoot: {Ts}) ***".format_map(locals()))
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
solver.solve([], [], [p0], 100)

print("step integration (guess): ", dt)
print("step integration (optimal): ", np.exp(solver.ps[0][0]))

# Printing the terminal pose
np.set_printoptions(precision=4, suppress=True)
print("Target pose:")
print("   position:", target_pos)
print("   quaternion:", target_quat.coeffs())
print("Terminal pose:")
print("   position:", solver.xs[-1][:3])
print("   quaternion:", solver.xs[-1][3:7])

if WITHPLOT:
    log = solver.getCallbacks()[1]
    crocoddyl.plotOCSolution(solver.xs, solver.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(
        log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=2
    )

if WITHDISPLAY:
    display = crocoddyl.MeshcatDisplay(hector)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)
