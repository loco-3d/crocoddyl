import os
import sys

import numpy as np
import pinocchio
from utils import import_control_example

import crocoddyl

WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ

oc = import_control_example("manipulator_fwddyn")
model = oc.kinova.model.copy()
state = crocoddyl.StateMultibody(model)

friction_joint_p = np.array(
    getattr(oc, "friction_parameters", [np.log(0.15), np.log(3.0), np.log(0.2)])
)
initial_friction_joint_p = np.log(0.7 * np.exp(friction_joint_p))
friction_joint_ids = list(range(1, state.pinocchio.njoints))
joint_dynamics = []
for jid in friction_joint_ids:
    joint = state.pinocchio.joints[jid]
    joint_dynamics.append(
        crocoddyl.JointDynamicsModelFriction(
            jid,
            joint.nq,
            friction_joint_p,
            crocoddyl.JointFrictionType.COULOMB_VISCOUS,
        )
    )
actuation = crocoddyl.ActuationModelMultibody(state, joint_dynamics)
np_actuation = actuation.np
groundtruth_actuation_p = np.tile(friction_joint_p, len(friction_joint_ids))
initial_actuation_p = np.tile(initial_friction_joint_p, len(friction_joint_ids))

nbodies = model.nbodies - 1
np_inertial = nbodies * 10
parametrization = crocoddyl.ExpEigenValueParametrization()
inertial_params = crocoddyl.MultibodyInertialParams(state, parametrization)
param = inertial_params.parametrization
param_scratch = param.createData()

groundtruth_inertial_p = np.ones(np_inertial)
initial_inertial_p = np.ones(np_inertial)
for i in range(nbodies):
    psi = model.inertias[i + 1].toDynamicParameters()
    s = slice(i * 10, (i + 1) * 10)
    param.toParametrization(groundtruth_inertial_p[s], psi)
    param.toParametrization(initial_inertial_p[s], 0.7 * psi)

for i in range(nbodies):
    psi = np.zeros(10)
    s = slice(i * 10, (i + 1) * 10)
    param.fromParametrization(param_scratch, psi, initial_inertial_p[s])
    model.inertias[i + 1] = pinocchio.Inertia.FromDynamicParameters(psi)

params = crocoddyl.ParameterManager(state)
params.addParam("actuation", crocoddyl.ActuationMultibodyParams(actuation))
params.addParam("inertial", inertial_params)

nu = state.ndx + state.nv
nv = state.nv
np_total = np_actuation + np_inertial
actuation_slice = slice(0, np_actuation)
inertial_slice = slice(np_actuation, np_total)
groundtruth_p = np.concatenate([groundtruth_actuation_p, groundtruth_inertial_p])
initial_p = np.concatenate([initial_actuation_p, initial_inertial_p])

pResidual = crocoddyl.ResidualModelParameters(state, initial_p, nu)
pRegCost = crocoddyl.CostModelResidual(state, pResidual)
wResidual = crocoddyl.ResidualModelControl(state, nu)
wActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([1.0] * state.ndx + [0.0] * nv)
)
wRegCost = crocoddyl.CostModelResidual(state, wActivation, wResidual)
xActivation = crocoddyl.ActivationModelWeightedQuad(np.array([1.0] * nv + [0.1] * nv))

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
        crocoddyl.ImplicitConstraintModelMultiple(state, nv),
        np_total,
        crocoddyl.DynamicsType.ContinuousEstimation,
    )
    runningModels.append(
        crocoddyl.IntegratedObserverModelEuler(dynamics, runningCostModel, None, oc.dt)
    )

terminalCostModel = crocoddyl.CostModelSum(state, nu, np_total)
xResidual = crocoddyl.ResidualModelState(state, oc.solver.xs[-1], nu)
xObsCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
terminalCostModel.addCost("xObs", xObsCost, 1e-1)
terminalDynamics = crocoddyl.DynamicsModelConstrainedInverse(
    state,
    actuation,
    crocoddyl.ImplicitConstraintModelMultiple(state, nv),
    np_total,
    crocoddyl.DynamicsType.ContinuousEstimation,
)
terminalModel = crocoddyl.IntegratedObserverModelEuler(
    terminalDynamics, terminalCostModel, None, oc.dt
)

problem = crocoddyl.ObservationProblem(
    oc.solver.xs[0],
    oc.solver.us,
    runningModels,
    terminalModel,
    crocoddyl.ParameterPhaseModel(params),
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

print("*** manipulator inertial parameter estimation (invdyn) ***")
solver.solve(init_xs=list(oc.solver.xs), init_us=[], init_p=[initial_p], maxiter=100)

p_est = solver.p[0]
print("\nEstimated vs. ground-truth inertial parameters:")
inertial_p_est = p_est[inertial_slice]
for i in range(nbodies):
    s = slice(i * 10, (i + 1) * 10)
    psi_est, psi_gt = np.zeros(10), np.zeros(10)
    param.fromParametrization(param_scratch, psi_est, inertial_p_est[s])
    param.fromParametrization(param_scratch, psi_gt, groundtruth_inertial_p[s])
    print(f"  body {i + 1}: mass est={psi_est[0]:.4f}  gt={psi_gt[0]:.4f}")

print("\nEstimated vs. ground-truth friction parameters:")
actuation_p_est = p_est[actuation_slice]
for i, jid in enumerate(friction_joint_ids):
    s = slice(i * len(friction_joint_p), (i + 1) * len(friction_joint_p))
    gamma_est = np.exp(actuation_p_est[s])
    gamma_gt = np.exp(groundtruth_actuation_p[s])
    print(
        f"  {state.pinocchio.names[jid]}: "
        f"coulomb est={gamma_est[0]:.4f} gt={gamma_gt[0]:.4f}, "
        f"sharpness est={gamma_est[1]:.4f} gt={gamma_gt[1]:.4f}, "
        f"viscous est={gamma_est[2]:.4f} gt={gamma_gt[2]:.4f}"
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
    )
