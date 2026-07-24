import os
import sys

import numpy as np
import pinocchio
from utils import (
    compute_inertial_covariances,
    import_control_example,
    plot_inertial_estimation,
)

import crocoddyl

WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ

oc = import_control_example("double_pendulum_continuous_fwddyn")

model = oc.pendulum.model.copy()
original_total_mass = pinocchio.computeTotalMass(model)
state = crocoddyl.StateMultibody(model)
actLink = 1
actuated_joint = state.pinocchio.joints[actLink]
actuation = crocoddyl.ActuationModelMultibody(
    state,
    [
        crocoddyl.JointDynamicsModelIdentity(
            actLink, actuated_joint.nq, actuated_joint.nv
        )
    ],
)

nbodies = model.nbodies - 1
np_inertial = nbodies * 10
parametrization = crocoddyl.ExpEigenValueParametrization()
inertial_params = crocoddyl.MultibodyInertialParams(state, parametrization)
param = inertial_params.parametrization
param_scratch = param.createData()

groundtruth_p = np.ones(np_inertial)
initial_p = np.ones(np_inertial)
for i in range(nbodies):
    psi = model.inertias[i + 1].toDynamicParameters()
    s = slice(i * 10, (i + 1) * 10)
    param.toParametrization(groundtruth_p[s], psi)
    param.toParametrization(initial_p[s], 0.9 * psi)

for i in range(nbodies):
    psi = np.zeros(10)
    s = slice(i * 10, (i + 1) * 10)
    param.fromParametrization(param_scratch, psi, initial_p[s])
    model.inertias[i + 1] = pinocchio.Inertia.FromDynamicParameters(psi)

params = crocoddyl.ParameterManager(state)
params.addParam("inertial", inertial_params)

nu = state.ndx + state.nv
nv = state.nv
np_total = np_inertial

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
    runningCostModel.addCost("pReg", pRegCost, 1e-2)
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
terminalCostModel.addCost("xObs", xObsCost, 1e1)
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
    oc.solver.xs[0], oc.solver.us, runningModels, terminalModel, params
)
problem.update_p(initial_p, phase_idx=0)

solver = crocoddyl.SolverIntro(
    problem,
    crocoddyl.FeasShoot,
    crocoddyl.LuNull,
    crocoddyl.LuNull,
    crocoddyl.AStateSchur,
)
if WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

print("*** double-pendulum inertial parameter estimation (invdyn) ***")
solver.solve(init_xs=list(oc.solver.xs), init_us=[], init_p=[initial_p], maxiter=100)

p_est = solver.p[0]
print("\nEstimated vs. ground-truth inertial parameters:")
for i in range(nbodies):
    s = slice(i * 10, (i + 1) * 10)
    psi_est, psi_gt = np.zeros(10), np.zeros(10)
    param.fromParametrization(param_scratch, psi_est, p_est[s])
    param.fromParametrization(param_scratch, psi_gt, groundtruth_p[s])
    print(f"  body {i + 1}: mass est={psi_est[0]:.4f}  gt={psi_gt[0]:.4f}")

if WITHPLOT:
    plot_inertial_estimation(
        solver,
        oc.solver,
        state,
        param,
        param_scratch,
        groundtruth_p,
        nbodies,
        compute_inertial_covariances(
            solver, param, param_scratch, nbodies, initial_p=initial_p
        ),
    )
