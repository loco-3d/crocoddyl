import os
import sys

from utils import (
    create_gait_estimation_problem,
    import_control_example,
    print_friction_estimation,
    print_inertial_estimation,
)

import crocoddyl

WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ

oc = import_control_example("biped_gaits_fwddyn")
control_solver = oc.solver[0]
estimation = create_gait_estimation_problem(
    oc.talos_legs.model,
    oc.gait,
    control_solver,
    fwddyn=True,
    parametrization=crocoddyl.ExpEigenValueParametrization(),
    friction_type=crocoddyl.JointFrictionType.COULOMB_VISCOUS,
)

solver = crocoddyl.SolverFDDP(
    estimation["problem"],
    crocoddyl.FeasShoot,
    crocoddyl.LuNull,
    crocoddyl.AStateQrNull,
)
if WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

print("*** biped gait friction and inertial parameter estimation (fwddyn) ***")
solver.solve(
    init_xs=list(control_solver.xs),
    init_us=[],
    init_ps=[estimation["initial_p"]],
    maxiter=100,
)

p_est = solver.ps[0]
print_inertial_estimation(
    estimation["state"],
    estimation["parametrization"],
    estimation["param_scratch"],
    p_est[estimation["inertial_slice"]],
    estimation["groundtruth_inertial_p"],
    estimation["nbodies"],
)
print_friction_estimation(
    estimation["state"],
    estimation["joint_ids"],
    estimation["friction_parameters"],
    p_est[estimation["actuation_slice"]],
    estimation["groundtruth_actuation_p"],
)

if WITHPLOT:
    crocoddyl.plotInertialEstimationWithCovariance(
        solver,
        control_solver,
        estimation["state"],
        estimation["parametrization"],
        estimation["param_scratch"],
        estimation["groundtruth_inertial_p"],
        estimation["nbodies"],
        crocoddyl.computeInertialCovariances(
            solver,
            estimation["parametrization"],
            estimation["param_scratch"],
            estimation["nbodies"],
            initial_p=estimation["initial_p"],
            parameter_slice=estimation["inertial_slice"],
        ),
        show=False,
    )

    nfriction = len(estimation["friction_parameters"])
    joint_names = [
        estimation["state"].pinocchio.names[jid] for jid in estimation["joint_ids"]
    ]
    crocoddyl.plotFrictionParam(
        p_est[estimation["actuation_slice"]].reshape(-1, nfriction),
        estimation["friction_type"],
        nominal=estimation["groundtruth_actuation_p"].reshape(-1, nfriction),
        figIndex=6 + estimation["nbodies"],
        figTitle="Friction model",
        joint_name=joint_names,
        parametrized=True,
    )
