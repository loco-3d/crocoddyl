import os
import sys

import numpy as np
import pinocchio
from utils import import_control_example

import crocoddyl

WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ

oc = import_control_example("b1_leg_6d_invdyn")

model = oc.model.copy()
state = crocoddyl.StateMultibody(model)

actuated_joints = ["FL_hip_joint", "FL_thigh_joint", "crank_bar2_thigh"]
friction_type = oc.friction_type
friction_parameters = np.array(oc.friction_parameters, copy=True)
initial_friction_parameters = np.log(0.7 * np.exp(friction_parameters))
joint_models = []
friction_joint_ids = []
for joint_name in actuated_joints:
    jid = model.getJointId(joint_name)
    joint = model.joints[jid]
    friction_joint_ids.append(jid)
    joint_models.append(
        crocoddyl.JointDynamicsModelFriction(
            jid, joint.nq, initial_friction_parameters, friction_type
        )
    )
actuation = crocoddyl.ActuationModelMultibody(state, joint_models)
np_actuation = actuation.np
groundtruth_actuation_p = np.tile(friction_parameters, len(friction_joint_ids))
initial_actuation_p = np.tile(initial_friction_parameters, len(friction_joint_ids))

constraint_nu = state.nv + sum(
    sum(bool(active) for active in cm.mask) for cm in oc.constraint_models
)
constraints = crocoddyl.ImplicitConstraintModelMultiple(state, constraint_nu)
for i, cm in enumerate(oc.constraint_models):
    constraints.addConstraint(
        f"loop_contact_{i}",
        crocoddyl.KinematicLoopModel(
            state,
            cm.joint1_id,
            cm.joint1_placement,
            cm.joint2_id,
            cm.joint2_placement,
            pinocchio.LOCAL,
            constraint_nu,
            np.array([1000.0, 10.0]),
            [bool(active) for active in cm.mask],
        ),
    )

body_names = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "link_1_crank_bar1",
    "crank_bar2_thigh",
    "link_2_crank_bar2",
]
nbodies = len(body_names)
np_inertial = nbodies * 10
parametrization = crocoddyl.ExpEigenValueParametrization()
inertial_params = crocoddyl.MultibodyInertialParams(state, parametrization, body_names)
param = inertial_params.parametrization
param_scratch = param.createData()

groundtruth_inertial_p = np.ones(np_inertial)
initial_inertial_p = np.ones(np_inertial)
for i, body_name in enumerate(body_names):
    jid = model.getJointId(body_name)
    psi = model.inertias[jid].toDynamicParameters()
    s = slice(i * 10, (i + 1) * 10)
    param.toParametrization(groundtruth_inertial_p[s], psi)
    param.toParametrization(initial_inertial_p[s], 0.7 * psi)

for i, body_name in enumerate(body_names):
    jid = model.getJointId(body_name)
    psi = np.zeros(10)
    s = slice(i * 10, (i + 1) * 10)
    param.fromParametrization(param_scratch, psi, initial_inertial_p[s])
    model.inertias[jid] = pinocchio.Inertia.FromDynamicParameters(psi)

params = crocoddyl.ParameterManager(state)
params.addParam("actuation", crocoddyl.ActuationMultibodyParams(actuation))
params.addParam("inertial", inertial_params)

dynamics_nu = constraint_nu
nu = state.ndx + dynamics_nu
np_total = np_actuation + np_inertial
actuation_slice = slice(0, np_actuation)
inertial_slice = slice(np_actuation, np_total)
groundtruth_p = np.concatenate([groundtruth_actuation_p, groundtruth_inertial_p])
initial_p = np.concatenate([initial_actuation_p, initial_inertial_p])
x_weights = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0])
xActivation = crocoddyl.ActivationModelWeightedQuad(x_weights)
pResidual = crocoddyl.ResidualModelParameters(state, initial_p, nu)
pRegCost = crocoddyl.CostModelResidual(state, pResidual)
wResidual = crocoddyl.ResidualModelControl(state, nu)
wActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([1.0] * state.ndx + [0.0] * dynamics_nu)
)
wRegCost = crocoddyl.CostModelResidual(state, wActivation, wResidual)

runningModels = []
for t in range(oc.T):
    runningCostModel = crocoddyl.CostModelSum(state, nu, np_total)
    runningCostModel.addCost("pReg", pRegCost, 1e-3)
    runningCostModel.addCost("wReg", wRegCost, 1e2)
    xResidual = crocoddyl.ResidualModelState(state, oc.solver.xs[t], nu)
    xObsCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    runningCostModel.addCost("xObs", xObsCost, 1e1)
    dynamics = crocoddyl.DynamicsModelConstrainedInverse(
        state,
        actuation,
        constraints,
        np_total,
        crocoddyl.DynamicsType.ContinuousEstimation,
    )
    runningModels.append(
        crocoddyl.IntegratedObserverModelEuler(dynamics, runningCostModel, None, oc.dt)
    )

terminalCostModel = crocoddyl.CostModelSum(state, nu, np_total)
xResidual = crocoddyl.ResidualModelState(state, oc.solver.xs[-1], nu)
xObsCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
terminalCostModel.addCost("xObs", xObsCost, 1e1)
terminalDynamics = crocoddyl.DynamicsModelConstrainedInverse(
    state,
    actuation,
    constraints,
    np_total,
    crocoddyl.DynamicsType.ContinuousEstimation,
)
terminalModel = crocoddyl.IntegratedObserverModelEuler(
    terminalDynamics, terminalCostModel, None, oc.dt
)

oc.solver.problem.calc(oc.solver.xs, oc.solver.us)
measured_taus = []
for data in oc.solver.problem.runningDatas:
    measured_taus.append(np.array(data.dynamics.multibody.joint.tau, copy=True))

problem = crocoddyl.ObservationProblem(
    oc.solver.xs[0], measured_taus, runningModels, terminalModel, params
)
problem.update_p(initial_p, phase_idx=0)

solver = crocoddyl.SolverIntro(
    problem,
    crocoddyl.FeasShoot,
    crocoddyl.LuNull,
    crocoddyl.LuNull,
    crocoddyl.AStateQrNull,
)
if WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

print("*** b1-leg 6D inertial parameter estimation (invdyn) ***")
solver.solve(init_xs=list(oc.solver.xs), init_us=[], init_p=[initial_p], maxiter=100)

p_est = solver.p[0]
print("\nEstimated vs. ground-truth inertial parameters:")
inertial_p_est = p_est[inertial_slice]
for i, body_name in enumerate(body_names):
    s = slice(i * 10, (i + 1) * 10)
    psi_est, psi_gt = np.zeros(10), np.zeros(10)
    param.fromParametrization(param_scratch, psi_est, inertial_p_est[s])
    param.fromParametrization(param_scratch, psi_gt, groundtruth_inertial_p[s])
    print(f"  {body_name}: mass est={psi_est[0]:.4f}  gt={psi_gt[0]:.4f}")

print("\nEstimated vs. ground-truth friction parameters:")
actuation_p_est = p_est[actuation_slice]
nfriction = len(friction_parameters)
for i, jid in enumerate(friction_joint_ids):
    s = slice(i * nfriction, (i + 1) * nfriction)
    gamma_est = np.exp(actuation_p_est[s])
    gamma_gt = np.exp(groundtruth_actuation_p[s])
    print(
        f"  {state.pinocchio.names[jid]}: "
        f"coulomb est={gamma_est[0]:.4f} gt={gamma_gt[0]:.4f}, "
        f"sharpness est={gamma_est[1]:.4f} gt={gamma_gt[1]:.4f}"
    )

if WITHPLOT:
    crocoddyl.plotInertialEstimationWithCovariance(
        solver,
        oc.solver,
        state,
        param,
        param_scratch,
        groundtruth_inertial_p,
        nbodies,
        crocoddyl.computeInertialCovariances(
            solver,
            param,
            param_scratch,
            nbodies,
            initial_p=initial_p,
            parameter_slice=inertial_slice,
        ),
        show=False,
    )
    crocoddyl.plotFrictionParam(
        p_est[actuation_slice].reshape(-1, nfriction),
        friction_type,
        nominal=groundtruth_actuation_p.reshape(-1, nfriction),
        figIndex=6 + nbodies,
        figTitle="Friction model",
        joint_name=actuated_joints,
        parametrized=True,
    )
