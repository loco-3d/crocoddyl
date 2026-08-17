import importlib
import os
import sys
from enum import Enum

import numpy as np
import pinocchio

import crocoddyl


class SymmetryAxis(Enum):
    X = "x"
    Y = "y"
    Z = "z"


def import_control_example(module_name):
    examples_dir = os.path.dirname(os.path.dirname(__file__))
    if examples_dir not in sys.path:
        sys.path.insert(0, examples_dir)

    argv = sys.argv[:]
    plot_env = os.environ.pop("CROCODDYL_PLOT", None)
    estimation_import_env = os.environ.get("CROCODDYL_ESTIMATION_IMPORT")
    os.environ["CROCODDYL_ESTIMATION_IMPORT"] = "1"
    sys.argv = [arg for arg in sys.argv if arg != "plot"]
    sys.stdout.flush()
    sys.stderr.flush()
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with open(os.devnull, "w") as devnull:
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            return importlib.import_module(module_name)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
            sys.argv = argv
            if plot_env is not None:
                os.environ["CROCODDYL_PLOT"] = plot_env
            if estimation_import_env is None:
                os.environ.pop("CROCODDYL_ESTIMATION_IMPORT", None)
            else:
                os.environ["CROCODDYL_ESTIMATION_IMPORT"] = estimation_import_env


def actuated_joint_ids(model):
    first_joint = (
        model.getJointId("root_joint") + 1 if model.existJointName("root_joint") else 1
    )
    return [
        jid for jid in range(first_joint, model.njoints) if model.joints[jid].nv == 1
    ]


def create_friction_actuation(state, friction_parameters, friction_type):
    root_joint_id = (
        state.pinocchio.getJointId("root_joint")
        if state.pinocchio.existJointName("root_joint")
        else 0
    )
    friction_joint_ids = set(actuated_joint_ids(state.pinocchio))
    joint_dynamics = []
    for jid in range(1, state.pinocchio.njoints):
        joint = state.pinocchio.joints[jid]
        if jid in friction_joint_ids:
            joint_dynamics.append(
                crocoddyl.JointDynamicsModelFriction(
                    jid, joint.nq, friction_parameters, friction_type
                )
            )
        elif jid != root_joint_id:
            joint_dynamics.append(
                crocoddyl.JointDynamicsModelIdentity(jid, joint.nq, joint.nv)
            )
    return crocoddyl.ActuationModelMultibody(state, joint_dynamics)


def create_friction_parameter_vectors(model, friction_parameters, initial_scale=0.7):
    joint_ids = actuated_joint_ids(model)
    groundtruth = np.tile(friction_parameters, len(joint_ids))
    initial = np.tile(
        np.log(initial_scale * np.exp(friction_parameters)), len(joint_ids)
    )
    return joint_ids, groundtruth, initial


def create_inertial_parameter_vectors(model, parametrization, initial_scale=0.9):
    nbodies = model.nbodies - 1
    nparams = nbodies * 10
    groundtruth = np.ones(nparams)
    initial = np.ones(nparams)
    scratch = parametrization.createData()
    for i in range(nbodies):
        psi = model.inertias[i + 1].toDynamicParameters()
        s = slice(i * 10, (i + 1) * 10)
        parametrization.toParametrization(groundtruth[s], psi)
        parametrization.toParametrization(initial[s], initial_scale * psi)
    for i in range(nbodies):
        psi = np.zeros(10)
        s = slice(i * 10, (i + 1) * 10)
        parametrization.fromParametrization(scratch, psi, initial[s])
        model.inertias[i + 1] = pinocchio.Inertia.FromDynamicParameters(psi)
    return nbodies, groundtruth, initial, scratch


def make_total_mass_feasible_initial_inertial_parameters(
    parametrization, scratch, initial_p, mass_ref, nbodies
):
    dynamic = np.zeros(nbodies * 10)
    for i in range(nbodies):
        s = slice(i * 10, (i + 1) * 10)
        parametrization.fromParametrization(scratch, dynamic[s], initial_p[s])

    mass_error = float(mass_ref - np.sum(dynamic[0::10]))
    if abs(mass_error) <= 1e-12:
        return np.array(initial_p, copy=True)

    projected = np.array(initial_p, copy=True)
    base = slice(0, 10)
    base_mass = dynamic[0]
    corrected_mass = base_mass + mass_error
    if corrected_mass <= 0.0:
        raise ValueError(
            "Cannot make initial inertial parameters total-mass feasible "
            "without creating a non-positive base mass."
        )
    dynamic[base] *= corrected_mass / base_mass
    parametrization.toParametrization(projected[base], dynamic[base])
    return projected


def copy_contact_constraints(state, source_constraints, nu):
    constraints = crocoddyl.ImplicitConstraintModelMultiple(state, nu)
    for name, item in source_constraints.constraints.todict().items():
        constraint = item.constraint
        if not isinstance(constraint, crocoddyl.ContactModel):
            raise TypeError(f"Unsupported implicit constraint type for {name}.")
        constraints.addConstraint(
            name,
            crocoddyl.ContactModel(
                state,
                constraint.id,
                constraint.reference,
                constraint.type,
                nu,
                constraint.gains,
                constraint.mask,
            ),
            item.active,
        )
    return constraints


def gait_state_observation_activation(state):
    njoint = state.nv - 6
    weights = np.array(
        [0.0] * 6 + [1.0] * njoint + [0.0] * 3 + [1.0] * 3 + [0.1] * njoint
    )
    return crocoddyl.ActivationModelWeightedQuad(weights)


def create_gait_observation_cost(state, xref, initial_p, nu, nparams, fwddyn):
    costs = crocoddyl.CostModelSum(state, nu, nparams)
    p_residual = crocoddyl.ResidualModelParameters(state, initial_p, nu)
    costs.addCost("pReg", crocoddyl.CostModelResidual(state, p_residual), 1e-4)

    w_residual = crocoddyl.ResidualModelControl(state, nu)
    if fwddyn:
        w_cost = crocoddyl.CostModelResidual(state, w_residual)
    else:
        w_activation = crocoddyl.ActivationModelWeightedQuad(
            np.array([1.0] * state.ndx + [0.0] * (nu - state.ndx))
        )
        w_cost = crocoddyl.CostModelResidual(state, w_activation, w_residual)
    costs.addCost("wReg", w_cost, 1e2)

    x_residual = crocoddyl.ResidualModelState(state, xref, nu)
    x_cost = crocoddyl.CostModelResidual(
        state, gait_state_observation_activation(state), x_residual
    )
    costs.addCost("xObs", x_cost, 1e1)
    return costs


def create_gait_terminal_cost(state, xref, nu, nparams):
    costs = crocoddyl.CostModelSum(state, nu, nparams)
    x_residual = crocoddyl.ResidualModelState(state, xref, nu)
    x_cost = crocoddyl.CostModelResidual(
        state, gait_state_observation_activation(state), x_residual
    )
    costs.addCost("xObs", x_cost, 1e1)
    return costs


def create_total_mass_parameter_constraint(state, mass_ref, nparams, nu):
    constraints = crocoddyl.ConstraintModelManager(state, nu, nparams)
    mass_residual = crocoddyl.ResidualModelTotalMass(
        state, mass_ref, nu, nparams, "inertial"
    )
    constraints.addConstraint(
        "total_mass",
        crocoddyl.ConstraintModelResidual(state, mass_residual),
    )
    return constraints


def build_inertial_symmetry_matrix(model, mirror_pairs, equal_pairs):
    num_bodies = model.nbodies - 1
    dim_p = 10 * num_bodies
    matrix = np.zeros((dim_p, dim_p))
    name_to_idx = {model.names[i]: i - 1 for i in range(1, model.nbodies)}

    for left, right, axis in mirror_pairs:
        if left not in name_to_idx or right not in name_to_idx:
            continue
        base_1 = 10 * name_to_idx[left]
        base_2 = 10 * name_to_idx[right]
        for i in range(10):
            row = np.zeros(dim_p)
            if i == 0:
                row[base_1 + i] = 1.0
                row[base_2 + i] = -1.0
            elif i == 1:
                row[base_1 + i] = -1.0 if axis == SymmetryAxis.X else 1.0
                row[base_2 + i] = 1.0
            elif i == 2:
                row[base_1 + i] = -1.0 if axis == SymmetryAxis.Y else 1.0
                row[base_2 + i] = 1.0
            elif i == 3:
                row[base_1 + i] = -1.0 if axis == SymmetryAxis.Z else 1.0
                row[base_2 + i] = 1.0
            elif i in (4, 5):
                row[base_1 + i] = 1.0
                row[base_2 + i] = 1.0 if axis == SymmetryAxis.Z else -1.0
            elif i == 6:
                row[base_1 + i] = 1.0
                row[base_2 + i] = -1.0
            else:
                continue
            matrix[base_1 + i, :] += row

    for body_1, body_2 in equal_pairs:
        if body_1 not in name_to_idx or body_2 not in name_to_idx:
            continue
        base_1 = 10 * name_to_idx[body_1]
        base_2 = 10 * name_to_idx[body_2]
        for i in range(7):
            row = np.zeros(dim_p)
            row[base_1 + i] = 1.0
            row[base_2 + i] = -1.0
            matrix[base_1 + i, :] += row
    return matrix


def build_friction_symmetry_matrix(model, joint_ids, equal_pairs, fdim):
    joint_names = [model.names[jid] for jid in joint_ids]
    name_to_idx = {name: i for i, name in enumerate(joint_names)}
    dim_f = fdim * len(joint_names)
    matrix = np.zeros((dim_f, dim_f))
    for joint_1, joint_2 in equal_pairs:
        if joint_1 not in name_to_idx or joint_2 not in name_to_idx:
            continue
        base_1 = fdim * name_to_idx[joint_1]
        base_2 = fdim * name_to_idx[joint_2]
        for i in range(fdim):
            row = np.zeros(dim_f)
            row[base_1 + i] = 1.0
            row[base_2 + i] = -1.0
            matrix[base_1 + i, :] += row
    return matrix


def create_parameter_constraint_manager(
    state,
    mass_ref,
    nparams,
    nu,
    inertial_symmetry=None,
    friction_symmetry=None,
):
    constraints = create_total_mass_parameter_constraint(state, mass_ref, nparams, nu)
    if inertial_symmetry is not None and np.any(inertial_symmetry):
        residual = crocoddyl.ResidualModelSymmetryParameters(
            state, inertial_symmetry, nu, nparams, "inertial"
        )
        constraints.addConstraint(
            "inertial_symmetry",
            crocoddyl.ConstraintModelResidual(state, residual),
        )
    if friction_symmetry is not None and np.any(friction_symmetry):
        residual = crocoddyl.ResidualModelSymmetryParameters(
            state, friction_symmetry, nu, nparams, "actuation"
        )
        constraints.addConstraint(
            "friction_symmetry",
            crocoddyl.ConstraintModelResidual(state, residual),
        )
    return constraints


def require_arrival_qp_parameter_constraints():
    if not hasattr(crocoddyl, "AStateQP"):
        raise RuntimeError(
            "This example requires Crocoddyl bindings with AStateQP and "
            "phase-level ObservationProblem parameter constraints. The "
            "currently imported crocoddyl package does not expose AStateQP; "
            "rebuild/install the updated C++/Python bindings with Odyn enabled."
        )


def parameter_constraint_violations(problem, state, p_vectors):
    if not hasattr(problem, "parameter_constraints"):
        return float("nan"), float("nan")

    max_eq = 0.0
    max_ineq = 0.0
    managers = list(problem.parameter_constraints)
    datas = list(problem.parameter_constraints_data)
    for phase_idx, (manager, data) in enumerate(zip(managers, datas)):
        if manager is None or data is None:
            continue
        if phase_idx < len(p_vectors):
            problem.update_p(np.asarray(p_vectors[phase_idx]), phase_idx=phase_idx)
        x0 = state.zero()
        u0 = np.zeros(int(manager.nu))
        manager.calc(data, x0, u0)
        manager.calcDiff(data, x0, u0)

        h = np.array(data.h, copy=True)
        if h.size:
            max_eq = max(max_eq, float(np.max(np.abs(h))))

        g = np.array(data.g, copy=True)
        if g.size:
            lower = np.array(manager.g_lb, copy=True)
            upper = np.array(manager.g_ub, copy=True)
            lower_violation = np.maximum(lower - g, 0.0)
            upper_violation = np.maximum(g - upper, 0.0)
            max_ineq = max(
                max_ineq,
                float(np.max(np.maximum(lower_violation, upper_violation))),
            )
    return max_eq, max_ineq


def create_gait_observer_model(
    control_model, state, actuation, xref, initial_p, nparams, fwddyn
):
    if isinstance(control_model.dynamics, crocoddyl.DynamicsModelImpulseForward):
        nu = state.ndx
        costs = create_gait_observation_cost(
            state, xref, initial_p, nu, nparams, fwddyn
        )
        dynamics = crocoddyl.DynamicsModelImpulseForward(
            state,
            copy_contact_constraints(state, control_model.dynamics.constraints, 0),
            nparams,
        )
        return crocoddyl.DiscretizedObserverModel(dynamics, costs, 0)

    source_constraints = control_model.dynamics.constraints
    if fwddyn:
        dynamics_nu = actuation.nu
        observer_nu = state.ndx
        dynamics_type = crocoddyl.DynamicsModelConstrainedForward
    else:
        dynamics_nu = state.nv + source_constraints.nc
        observer_nu = state.ndx + dynamics_nu
        dynamics_type = crocoddyl.DynamicsModelConstrainedInverse

    constraints = copy_contact_constraints(state, source_constraints, dynamics_nu)
    costs = create_gait_observation_cost(
        state, xref, initial_p, observer_nu, nparams, fwddyn
    )
    dynamics = dynamics_type(
        state,
        actuation,
        constraints,
        nparams,
        crocoddyl.DynamicsType.ContinuousEstimation,
    )
    return crocoddyl.IntegratedObserverModelEuler(
        dynamics,
        costs,
        None,
        control_model.integrator_time.timeStep,
    )


def create_gait_terminal_observer_model(
    control_model, state, actuation, xref, nparams, fwddyn
):
    if isinstance(control_model.dynamics, crocoddyl.DynamicsModelImpulseForward):
        nu = state.ndx
        costs = create_gait_terminal_cost(state, xref, nu, nparams)
        dynamics = crocoddyl.DynamicsModelImpulseForward(
            state,
            copy_contact_constraints(state, control_model.dynamics.constraints, 0),
            nparams,
        )
        return crocoddyl.DiscretizedObserverModel(dynamics, costs, 0)

    source_constraints = control_model.dynamics.constraints
    if fwddyn:
        dynamics_nu = actuation.nu
        observer_nu = state.ndx
        dynamics_type = crocoddyl.DynamicsModelConstrainedForward
    else:
        dynamics_nu = state.nv + source_constraints.nc
        observer_nu = state.ndx + dynamics_nu
        dynamics_type = crocoddyl.DynamicsModelConstrainedInverse

    constraints = copy_contact_constraints(state, source_constraints, dynamics_nu)
    costs = create_gait_terminal_cost(state, xref, observer_nu, nparams)
    dynamics = dynamics_type(
        state,
        actuation,
        constraints,
        nparams,
        crocoddyl.DynamicsType.ContinuousEstimation,
    )
    return crocoddyl.IntegratedObserverModelEuler(
        dynamics,
        costs,
        None,
        control_model.integrator_time.timeStep,
    )


def gait_measured_torques(control_solver, fwddyn):
    if fwddyn:
        return control_solver.us

    control_solver.problem.calc(control_solver.xs, control_solver.us)
    taus = []
    for model, data in zip(
        control_solver.problem.runningModels, control_solver.problem.runningDatas
    ):
        if isinstance(model.dynamics, crocoddyl.DynamicsModelImpulseForward):
            taus.append(np.zeros(0))
        else:
            taus.append(np.array(data.dynamics.multibody.joint.tau, copy=True))
    return taus


def create_gait_estimation_problem(
    model,
    gait,
    control_solver,
    fwddyn,
    parametrization=None,
    friction_type=crocoddyl.JointFrictionType.COULOMB_VISCOUS,
    friction_parameters=None,
    initial_inertial_scale=0.9,
    initial_friction_scale=0.7,
    enforce_total_mass_constraint=False,
):
    if enforce_total_mass_constraint:
        require_arrival_qp_parameter_constraints()

    estimation_model = model.copy()
    original_total_mass = pinocchio.computeTotalMass(estimation_model)
    state = crocoddyl.StateMultibody(estimation_model)
    parametrization = parametrization or crocoddyl.ExpEigenValueParametrization()
    inertial_params = crocoddyl.MultibodyInertialParams(state, parametrization)
    param = inertial_params.parametrization
    nbodies, groundtruth_inertial_p, initial_inertial_p, param_scratch = (
        create_inertial_parameter_vectors(
            estimation_model, param, initial_scale=initial_inertial_scale
        )
    )

    if gait.friction_type != friction_type:
        raise ValueError(
            "The control and estimation problems must use the same friction model"
        )
    control_friction_parameters = np.asarray(gait.friction_parameters, dtype=float)
    friction_parameters = (
        np.array(control_friction_parameters, copy=True)
        if friction_parameters is None
        else np.asarray(friction_parameters, dtype=float)
    )
    if not np.array_equal(friction_parameters, control_friction_parameters):
        raise ValueError(
            "The control and estimation problems must use the same friction parameters"
        )
    actuation = create_friction_actuation(state, friction_parameters, friction_type)
    joint_ids, groundtruth_actuation_p, initial_actuation_p = (
        create_friction_parameter_vectors(
            state.pinocchio,
            friction_parameters,
            initial_scale=initial_friction_scale,
        )
    )

    n_actuation = actuation.np
    n_inertial = nbodies * 10
    nparams = n_actuation + n_inertial
    actuation_slice = slice(0, n_actuation)
    inertial_slice = slice(n_actuation, nparams)
    groundtruth_p = np.concatenate([groundtruth_actuation_p, groundtruth_inertial_p])
    initial_p = np.concatenate([initial_actuation_p, initial_inertial_p])

    params = crocoddyl.ParameterManager(state)
    params.addParam("actuation", crocoddyl.ActuationMultibodyParams(actuation))
    params.addParam("inertial", inertial_params)
    parameter_constraints = None

    running_models = [
        create_gait_observer_model(
            control_model,
            state,
            actuation,
            control_solver.xs[t],
            initial_p,
            nparams,
            fwddyn,
        )
        for t, control_model in enumerate(control_solver.problem.runningModels)
    ]
    terminal_model = create_gait_terminal_observer_model(
        control_solver.problem.terminalModel,
        state,
        actuation,
        control_solver.xs[-1],
        nparams,
        fwddyn,
    )
    if enforce_total_mass_constraint:
        parameter_constraints = create_total_mass_parameter_constraint(
            state, original_total_mass, nparams, running_models[0].nu
        )
    tau_meas = gait_measured_torques(control_solver, fwddyn)
    try:
        if parameter_constraints is None:
            problem = crocoddyl.ObservationProblem(
                control_solver.xs[0],
                tau_meas,
                running_models,
                terminal_model,
                params,
            )
        else:
            problem = crocoddyl.ObservationProblem(
                control_solver.xs[0],
                tau_meas,
                running_models,
                terminal_model,
                params,
                parameter_constraints,
            )
    except Exception as exc:
        if parameter_constraints is not None:
            raise RuntimeError(
                "The imported crocoddyl bindings do not accept phase-level "
                "parameter constraints in ObservationProblem. Rebuild/install "
                "the updated bindings before running the constrained "
                "quadruped examples."
            ) from exc
        raise
    problem.update_p(initial_p, phase_idx=0)

    return {
        "problem": problem,
        "state": state,
        "parametrization": param,
        "param_scratch": param_scratch,
        "nbodies": nbodies,
        "joint_ids": joint_ids,
        "friction_parameters": friction_parameters,
        "friction_type": friction_type,
        "groundtruth_p": groundtruth_p,
        "initial_p": initial_p,
        "actuation_slice": actuation_slice,
        "inertial_slice": inertial_slice,
        "original_total_mass": original_total_mass,
        "parameter_constraints": parameter_constraints,
        "groundtruth_actuation_p": groundtruth_actuation_p,
        "groundtruth_inertial_p": groundtruth_inertial_p,
    }


def print_inertial_estimation(state, parametrization, scratch, p_est, p_gt, nbodies):
    print("\nEstimated vs. ground-truth inertial parameters:")
    for i in range(nbodies):
        s = slice(i * 10, (i + 1) * 10)
        psi_est, psi_gt = np.zeros(10), np.zeros(10)
        parametrization.fromParametrization(scratch, psi_est, p_est[s])
        parametrization.fromParametrization(scratch, psi_gt, p_gt[s])
        print(f"  body {i + 1}: mass est={psi_est[0]:.4f}  gt={psi_gt[0]:.4f}")


def print_friction_estimation(state, joint_ids, friction_parameters, p_est, p_gt):
    print("\nEstimated vs. ground-truth friction parameters:")
    nfriction = len(friction_parameters)
    for i, jid in enumerate(joint_ids):
        s = slice(i * nfriction, (i + 1) * nfriction)
        gamma_est = np.exp(p_est[s])
        gamma_gt = np.exp(p_gt[s])
        print(
            f"  {state.pinocchio.names[jid]}: "
            f"coulomb est={gamma_est[0]:.4f} gt={gamma_gt[0]:.4f}, "
            f"sharpness est={gamma_est[1]:.4f} gt={gamma_gt[1]:.4f}, "
            f"viscous est={gamma_est[2]:.4f} gt={gamma_gt[2]:.4f}"
        )


def _logged_phase_vector(entry):
    if len(entry) > 0 and np.isscalar(entry[0]):
        return np.asarray(entry, dtype=float)
    return np.asarray(entry[0], dtype=float)


def _logged_phase_matrix(entry):
    if entry is None:
        return None
    if isinstance(entry, (list, tuple)):
        return None if len(entry) == 0 else np.asarray(entry[0], dtype=float)
    matrix = np.asarray(entry, dtype=float)
    if matrix.ndim == 3 and matrix.shape[0] > 0:
        return matrix[0]
    return matrix if matrix.ndim == 2 else None


def _align_log(entries, size, fill=None):
    entries = list(entries)
    if len(entries) < size:
        return [fill] * (size - len(entries)) + entries
    return entries[-size:]


def _physical_inertial_parameters(parametrization, data, p, nbodies):
    values = np.zeros(nbodies * 10)
    jacobian = np.zeros((nbodies * 10, nbodies * 10))
    for body in range(nbodies):
        body_slice = slice(body * 10, (body + 1) * 10)
        p_body = np.asarray(p[body_slice], dtype=float)
        dynamic = np.zeros(10)
        parametrization.fromParametrization(data, dynamic, p_body)
        dynamic_jacobian = np.zeros((10, 10))
        parametrization.updateParametrizationDerivative(
            data, dynamic_jacobian, p_body, dynamic
        )

        mass = dynamic[0]
        values[body_slice] = dynamic
        values[body_slice][1:4] /= mass
        physical_jacobian = np.eye(10)
        physical_jacobian[1:4, :] = 0.0
        physical_jacobian[1:4, 0] = -dynamic[1:4] / mass**2
        physical_jacobian[1:4, 1:4] = np.eye(3) / mass
        jacobian[body_slice, body_slice] = physical_jacobian @ dynamic_jacobian
    return values, jacobian


def _initial_parameter_precision(solver, initial_p):
    problem = solver.problem
    if len(problem.runningModels) == 0:
        return None

    accepted_p = [np.array(p, dtype=float, copy=True) for p in solver.p]
    model = problem.runningModels[0]
    data = problem.runningDatas[0]
    x = np.asarray(solver.xs[0], dtype=float)
    u = np.asarray(solver.us[0], dtype=float)
    try:
        problem.update_p(np.asarray(initial_p, dtype=float), phase_idx=0)
        model.calc(data, x, u)
        model.calcDiff(data, x, u)
        return np.array(data.Lpp, dtype=float, copy=True)
    except (AttributeError, RuntimeError, ValueError):
        return None
    finally:
        for phase, p in enumerate(accepted_p):
            problem.update_p(p, phase_idx=phase)
        model.calc(data, x, u)
        model.calcDiff(data, x, u)


def compute_inertial_covariances(
    solver,
    parametrization,
    parametrization_data,
    nbodies,
    initial_p=None,
    parameter_slice=None,
    eigenvalue_floor=1e-12,
):
    """Map the solver parameter precision to physical inertial covariance."""
    log = solver.getCallbacks()[-1]
    p_entries = getattr(log, "p", [])
    if len(p_entries) == 0:
        p_entries = [[solver.p[0]]]
    p_log = [_logged_phase_vector(entry) for entry in p_entries]
    precision_entries = getattr(log, "Vpp_phase", [])
    if len(precision_entries) == 0:
        precision_entries = [solver.Vpp_phase]
    precision_log = _align_log(
        [_logged_phase_matrix(entry) for entry in precision_entries], len(p_log)
    )
    regularization_log = _align_log(log.pregs, len(p_log))
    x0 = 0

    if initial_p is not None:
        p_log.insert(0, np.asarray(initial_p, dtype=float))
        precision_log.insert(0, _initial_parameter_precision(solver, initial_p))
        regularization_log.insert(0, None)
        x0 = -1

    parameters_log = []
    standard_deviation_log = []
    for p, precision, regularization in zip(p_log, precision_log, regularization_log):
        if parameter_slice is not None:
            indices = np.arange(p.size)[parameter_slice]
            p = p[indices]
            if precision is not None:
                precision = precision[np.ix_(indices, indices)]

        parameters, dparameters_dp = _physical_inertial_parameters(
            parametrization, parametrization_data, p, nbodies
        )
        standard_deviation = np.full(nbodies * 10, np.nan)
        if precision is not None:
            precision = 0.5 * (precision + precision.T)
            if regularization is not None and regularization > 0.0:
                precision -= regularization * np.eye(precision.shape[0])
            eigenvalues, eigenvectors = np.linalg.eigh(precision)
            identified = eigenvalues > eigenvalue_floor
            inverse_eigenvalues = np.zeros_like(eigenvalues)
            inverse_eigenvalues[identified] = 1.0 / eigenvalues[identified]
            covariance = (eigenvectors * inverse_eigenvalues) @ eigenvectors.T
            variances = np.diag(dparameters_dp @ covariance @ dparameters_dp.T)
            standard_deviation = np.sqrt(np.maximum(variances, 0.0))

        parameters_log.append(parameters)
        standard_deviation_log.append(standard_deviation)

    return {
        "parameters": np.asarray(parameters_log).T,
        "standard_deviations": np.asarray(standard_deviation_log).T,
        "x0": x0,
    }


def _plot_parameter_grid(
    figure_index, values, standard_deviations, nominal, titles, columns, ylabel, x0
):
    import math

    import matplotlib.pyplot as plt

    rows = math.ceil(len(values) / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        num=figure_index,
        figsize=(5 * columns, 3.5 * rows),
        squeeze=False,
    )
    for index, (value, standard_deviation, reference, title) in enumerate(
        zip(values, standard_deviations, nominal, titles)
    ):
        axis = axes.flat[index]
        steps = np.arange(value.size) + x0
        axis.plot(steps, value, "b--", label="Estimation")
        axis.axhline(reference, color="r", label="True value")
        finite = np.isfinite(standard_deviation)
        if np.any(finite):
            width = 2.0 * standard_deviation[finite]
            axis.fill_between(
                steps[finite],
                value[finite] - width,
                value[finite] + width,
                color="tab:orange",
                alpha=0.2,
                label="+/- 2 std",
            )
        axis.set_title(title)
        axis.grid(True)
        axis.legend()
    for axis in axes.flat[len(values) :]:
        axis.set_visible(False)
    figure.supxlabel("Estimation iterations")
    figure.supylabel(ylabel)
    figure.tight_layout()


def plot_inertial_estimation(
    solver,
    control_solver,
    state,
    parametrization,
    parametrization_data,
    groundtruth_p,
    nbodies,
    covariance,
    show=True,
):
    """Plot trajectories and physical inertial-parameter estimates."""
    import matplotlib.pyplot as plt

    log = solver.getCallbacks()[-1]
    figure, axes = plt.subplots(2, 1, num=1)
    axes[0].plot(np.asarray(solver.xs))
    axes[0].set_title("State trajectory")
    axes[0].grid(True)
    max_control_dimension = max((u.size for u in solver.us), default=0)
    controls = np.full((len(solver.us), max_control_dimension), np.nan)
    for knot, control in enumerate(solver.us):
        controls[knot, : control.size] = control
    axes[1].plot(controls)
    axes[1].set_title("Control trajectory")
    axes[1].set_xlabel("Knots")
    axes[1].grid(True)
    figure.tight_layout()

    errors = [state.diff(x, x_ref) for x, x_ref in zip(log.xs, control_solver.xs)]
    figure, axis = plt.subplots(num=2)
    axis.plot(np.asarray(errors))
    axis.set_title("Estimation error")
    axis.set_xlabel("Knots")
    axis.grid(True)

    crocoddyl.plotConvergence(
        log.costs,
        log.pregs,
        log.dregs,
        log.grads,
        log.stops,
        log.steps,
        figIndex=3,
        show=False,
    )

    parameters = covariance["parameters"]
    standard_deviations = covariance["standard_deviations"]
    groundtruth = np.zeros(nbodies * 10)
    for body in range(nbodies):
        body_slice = slice(body * 10, (body + 1) * 10)
        parametrization.fromParametrization(
            parametrization_data, groundtruth[body_slice], groundtruth_p[body_slice]
        )
        groundtruth[body_slice][1:4] /= groundtruth[body_slice][0]

    mass_indices = np.arange(0, nbodies * 10, 10)
    _plot_parameter_grid(
        4,
        parameters[mass_indices],
        standard_deviations[mass_indices],
        groundtruth[mass_indices],
        [f"Body {body + 1}" for body in range(nbodies)],
        min(3, nbodies),
        "Mass",
        covariance["x0"],
    )

    com_indices = np.concatenate(
        [np.arange(body * 10 + 1, body * 10 + 4) for body in range(nbodies)]
    )
    _plot_parameter_grid(
        5,
        parameters[com_indices],
        standard_deviations[com_indices],
        groundtruth[com_indices],
        [f"Body {body + 1} - COM({axis})" for body in range(nbodies) for axis in "xyz"],
        3,
        "Center of mass",
        covariance["x0"],
    )

    inertia_names = ("Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz")
    for body in range(nbodies):
        indices = np.arange(body * 10 + 4, body * 10 + 10)
        _plot_parameter_grid(
            6 + body,
            parameters[indices],
            standard_deviations[indices],
            groundtruth[indices],
            inertia_names,
            3,
            f"Body {body + 1} inertia",
            covariance["x0"],
        )

    if show:
        plt.show()


def _friction_torque(velocity, parameters, friction_type):
    gamma = np.exp(parameters)
    if friction_type == crocoddyl.JointFrictionType.COULOMB:
        return gamma[0] * np.tanh(gamma[1] * velocity)
    if friction_type == crocoddyl.JointFrictionType.COULOMB_VISCOUS:
        return gamma[0] * np.tanh(gamma[1] * velocity) + gamma[2] * velocity
    raise ValueError("This example supports Coulomb and Coulomb-viscous friction.")


def plot_friction_parameters(
    parameters,
    friction_type,
    nominal,
    figure_index=1,
    title="Friction model",
    joint_names=None,
    show=True,
):
    """Plot estimated and nominal friction torque curves."""
    import math

    import matplotlib.pyplot as plt

    parameters = np.atleast_2d(np.asarray(parameters, dtype=float))
    nominal = np.atleast_2d(np.asarray(nominal, dtype=float))
    if nominal.shape != parameters.shape:
        raise ValueError("nominal and parameters must have the same shape.")

    columns = max(1, int(math.sqrt(parameters.shape[0])))
    rows = math.ceil(parameters.shape[0] / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        num=figure_index,
        figsize=(5 * columns, 3.5 * rows),
        squeeze=False,
    )
    velocity = np.linspace(-2.0, 2.0, 400)
    for index, parameter in enumerate(parameters):
        axis = axes.flat[index]
        axis.plot(
            velocity,
            _friction_torque(velocity, nominal[index], friction_type),
            "r-",
            label="True value",
        )
        axis.plot(
            velocity,
            _friction_torque(velocity, parameter, friction_type),
            "b--",
            label="Estimation",
        )
        axis.set_title(joint_names[index] if joint_names else f"Joint {index + 1}")
        axis.set_xlabel("Velocity [rad/s]")
        axis.set_ylabel("Friction torque [Nm]")
        axis.grid(True)
        axis.legend()
    for axis in axes.flat[parameters.shape[0] :]:
        axis.set_visible(False)
    figure.suptitle(title)
    figure.tight_layout()
    if show:
        plt.show()
