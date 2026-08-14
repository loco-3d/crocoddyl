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

# Load the B1 closed-chain leg model
robot = example_robot_data.load("b1_leg_6D")
constraint_models = robot.get_constraints()
q0 = robot.q0.copy()
model = robot.model
data = model.createData()
state = crocoddyl.StateMultibody(model)

# Creating the actuation model
actuated_joints = ["FL_hip_joint", "FL_thigh_joint", "crank_bar2_thigh"]
friction_type = crocoddyl.JointFrictionType.COULOMB
friction_gamma = np.array([2.0, 10.4])
friction_parameters = np.log(friction_gamma)
joint_models = []
for joint_name in actuated_joints:
    jid = model.getJointId(joint_name)
    joint = model.joints[jid]
    joint_models.append(
        crocoddyl.JointDynamicsModelFriction(
            jid, joint.nq, friction_parameters, friction_type
        )
    )
actuation = crocoddyl.ActuationModelMultibody(state, joint_models)
nc = sum(sum(bool(active) for active in cm.mask) for cm in constraint_models)
nu = state.nv + nc

# Creating the 6D loop constraints
constraints = crocoddyl.ImplicitConstraintModelMultiple(state, nu)
for i, cm in enumerate(constraint_models):
    constraints.addConstraint(
        f"loop_contact_{i}",
        crocoddyl.KinematicLoopModel(
            state,
            cm.joint1_id,
            cm.joint1_placement,
            cm.joint2_id,
            cm.joint2_placement,
            pinocchio.LOCAL,
            nu,
            np.array([1000.0, 10.0]),
            [bool(active) for active in cm.mask],
        ),
    )

# Defining the circular foot tracking task
T = 100
dt = 1e-2
x0 = np.concatenate([q0, np.zeros(model.nv)])
frame_name = "FL_foot"
frame_id = model.getFrameId(frame_name)
pinocchio.forwardKinematics(model, data, q0)
pinocchio.updateFramePlacements(model, data)
p0 = data.oMf[frame_id].translation.copy()
radius = 0.1
omega = 2.0 * np.pi * 3.0
circular_traj = [
    p0
    + np.array(
        [
            -radius * (np.cos(omega * t * dt) - 1.0),
            0.0,
            -radius * np.sin(omega * t * dt),
        ]
    )
    for t in range(T)
]

# Defining residuals and costs
xResidual = crocoddyl.ResidualModelState(state, x0, nu)
xActivation = crocoddyl.ActivationModelQuad(state.ndx)
uResidual = crocoddyl.ResidualModelJointEffort(
    state, actuation, np.zeros(actuation.nu), nu, False
)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
footActivation = crocoddyl.ActivationModelQuad(3)

# Creating the running and terminal models
runningModels = []
for t in range(T):
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    runningCostModel.addCost("uReg", uRegCost, 1e-4 / dt)
    runningCostModel.addCost("xReg", xRegCost, 1e-5 / dt)
    footResidual = crocoddyl.ResidualModelFrameTranslation(
        state, frame_id, circular_traj[t], nu
    )
    footCost = crocoddyl.CostModelResidual(state, footActivation, footResidual)
    runningCostModel.addCost("footTrack", footCost, 1e3)
    runningDynamics = crocoddyl.DynamicsModelConstrainedInverse(
        state, actuation, constraints
    )
    runningModels.append(
        crocoddyl.IntegratedActionModelEuler(
            runningDynamics,
            runningCostModel,
            None,
            crocoddyl.IntegratorTime(dt, False),
        )
    )

terminalCostModel = crocoddyl.CostModelSum(state, nu)
terminalResidual = crocoddyl.ResidualModelFrameTranslation(
    state, frame_id, circular_traj[-1], nu
)
terminalCostModel.addCost(
    "footTrack",
    crocoddyl.CostModelResidual(state, footActivation, terminalResidual),
    1e3,
)
terminalDynamics = crocoddyl.DynamicsModelConstrainedInverse(
    state, actuation, constraints
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    terminalDynamics,
    terminalCostModel,
    None,
    crocoddyl.IntegratorTime(0.0, False),
)

# Creating the shooting problem and the OC solver
problem = crocoddyl.ShootingProblem(x0, runningModels, terminalModel)
solver = crocoddyl.SolverIntro(problem)
if WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem
xs = [problem.x0] * (T + 1)
us = problem.quasiStatic(xs[:-1])
print("*** b1-leg 6D loop control (invdyn) ***")
solver.solve(xs, us, maxiter=50)

# Printing the terminal foot position
pinocchio.forwardKinematics(state.pinocchio, data, solver.xs[-1][: state.nq])
pinocchio.updateFramePlacements(state.pinocchio, data)
target = circular_traj[-1]
reached = data.oMf[frame_id].translation
np.set_printoptions(precision=4, suppress=True)
print("Target foot position:", target)
print("Reached foot position:", reached)
print("Foot tracking error:", np.linalg.norm(reached - target))

# Plotting the solution and the solver convergence
if WITHPLOT:
    log = solver.getCallbacks()[1]
    crocoddyl.plotOCSolution(solver.xs, solver.us, figIndex=1, show=False)
    crocoddyl.plotConvergence(
        log.costs,
        log.pregs,
        log.dregs,
        log.grads,
        log.stops,
        log.steps,
        figIndex=2,
    )

# Visualizing the solution in meshcat
if WITHDISPLAY:
    display = crocoddyl.MeshcatDisplay(robot)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)
