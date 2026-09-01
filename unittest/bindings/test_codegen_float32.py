###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################

import os
import tempfile
import unittest

import numpy as np
import pinocchio
import pinocchio.cppadcg as pinocg64
import pinocchio.cppadcg_float32 as pinocg32

try:
    import pinocchio.float32

    PINOCCHIO_FLOAT32_AVAILABLE = True
except ModuleNotFoundError:
    PINOCCHIO_FLOAT32_AVAILABLE = False

import crocoddyl
import crocoddyl.cgfloat32 as crocoddylcg32
import crocoddyl.cgfloat64 as crocoddylcg64
import crocoddyl.float32 as crocoddyl32


class CodeGenFloat32Test(unittest.TestCase):
    @staticmethod
    def assert_fields_close(test, lhs, rhs, fields, rtol, atol):
        for field in fields:
            np.testing.assert_allclose(
                getattr(lhs, field),
                getattr(rhs, field),
                rtol=rtol,
                atol=atol,
                err_msg=field,
            )

    def test_scalar_modules_and_casts(self):
        self.assertIs(crocoddyl.DType.values[0], crocoddyl.DType.Float64)
        self.assertIs(crocoddyl.DType.values[1], crocoddyl.DType.Float32)
        self.assertIs(crocoddyl.DType.values[2], crocoddyl.DType.ADFloat64)
        self.assertIs(crocoddyl.DType.values[3], crocoddyl.DType.ADFloat32)

        self.assertIsNot(crocoddyl.ActionModelLQR, crocoddyl32.ActionModelLQR)
        self.assertIsNot(crocoddylcg64.ActionModelLQR, crocoddylcg32.ActionModelLQR)
        self.assertIsNot(pinocg64.Model, pinocg32.Model)

        model64 = crocoddyl.ActionModelLQR(4, 2)
        model32 = model64.cast(crocoddyl.DType.Float32)
        model_ad64 = model32.cast(crocoddyl.DType.ADFloat64)
        model_ad32 = model32.cast(crocoddyl.DType.ADFloat32)

        self.assertIsInstance(model32, crocoddyl32.ActionModelLQR)
        self.assertIsInstance(model_ad64, crocoddylcg64.ActionModelLQR)
        self.assertIsInstance(model_ad32, crocoddylcg32.ActionModelLQR)
        self.assertIsInstance(
            model_ad64.cast(crocoddyl.DType.ADFloat32),
            crocoddylcg32.ActionModelLQR,
        )
        self.assertIsInstance(
            model_ad32.cast(crocoddyl.DType.ADFloat64),
            crocoddylcg64.ActionModelLQR,
        )

        if PINOCCHIO_FLOAT32_AVAILABLE:
            state64 = crocoddyl.StateMultibody(pinocchio.buildSampleModelManipulator())
            state32 = state64.cast(crocoddyl.DType.Float32)
            state_ad32 = state32.cast(crocoddyl.DType.ADFloat32)
            self.assertIsInstance(state32, crocoddyl32.StateMultibody)
            self.assertIsInstance(state_ad32, crocoddylcg32.StateMultibody)
            self.assertIsInstance(state_ad32.pinocchio, pinocg32.Model)

    def test_parameterized_action_codegen(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                for module, dtype, suffix, rtol, atol in (
                    (crocoddyl, np.float64, "float64", 1e-9, 1e-10),
                    (crocoddyl32, np.float32, "float32", 2e-4, 2e-5),
                ):
                    with self.subTest(module=module.__name__):
                        model = module.ActionModelLQR(4, 3, 2, 2, 1, False)
                        manager = module.ParameterManager(model.state)
                        manager.addParam("lqr", module.LQRParams(model.state, 2))
                        manager.addParam(
                            "inactive",
                            module.LQRParams(model.state, 1),
                            False,
                        )
                        data = model.createData()
                        model.set_params(data, manager)
                        p = np.array([0.17, -0.09], dtype=dtype)
                        model.update_p(data, p)
                        x = np.linspace(-0.2, 0.2, model.state.nx, dtype=dtype)
                        u = np.linspace(-0.1, 0.1, model.nu, dtype=dtype)
                        model.calc(data, x, u)
                        model.calcDiff(data, x, u)

                        lib_name = "crocoddyl_lqr_" + suffix
                        generated = module.ActionModelCodeGen(model, lib_name)
                        generated_data = generated.createData()
                        generated.set_params(generated_data, manager)
                        generated.update_p(generated_data, p)
                        generated.calc(generated_data, x, u)
                        generated.calcDiff(generated_data, x, u)

                        self.assertTrue(generated.existLib(lib_name))
                        self.assertEqual(generated.np, 2)
                        self.assertEqual(generated_data.xnext.dtype, dtype)
                        self.assert_fields_close(
                            self,
                            generated_data,
                            data,
                            (
                                "xnext",
                                "r",
                                "g",
                                "h",
                                "Fx",
                                "Fu",
                                "Fp",
                                "Lx",
                                "Lu",
                                "Lp",
                                "Lxx",
                                "Lxu",
                                "Luu",
                                "Lpp",
                                "Lpx",
                                "Lpu",
                                "Gx",
                                "Gu",
                                "Gp",
                                "Hx",
                                "Hu",
                                "Hp",
                            ),
                            rtol,
                            atol,
                        )
                        self.assertGreater(
                            sum(
                                np.linalg.norm(getattr(generated_data, field))
                                for field in ("Fp", "Lp", "Lpp", "Lpx", "Lpu")
                            ),
                            0.0,
                        )
                        self.assertAlmostEqual(
                            generated_data.cost,
                            data.cost,
                            delta=max(atol, abs(data.cost) * rtol),
                        )

                        loaded = module.ActionModelCodeGen(lib_name, model)
                        loaded_data = loaded.createData()
                        loaded.set_params(loaded_data, manager)
                        loaded.update_p(loaded_data, p)
                        loaded.calc(loaded_data, x, u)
                        loaded.calcDiff(loaded_data, x, u)
                        self.assert_fields_close(
                            self,
                            loaded_data,
                            generated_data,
                            ("xnext", "Fp", "Lp", "Lpp", "Gp", "Hp"),
                            rtol,
                            atol,
                        )

                        terminal_data = generated.createData()
                        generated.set_params(terminal_data, manager)
                        generated.update_p(terminal_data, p)
                        sentinels = {
                            field: np.full(
                                getattr(terminal_data, field).shape,
                                30 + index,
                                dtype=dtype,
                            )
                            for index, field in enumerate(
                                ("Fp", "Fu", "Lu", "Lxu", "Luu", "Lpu", "Gu", "Hu")
                            )
                        }
                        for field, value in sentinels.items():
                            getattr(terminal_data, field)[:] = value
                        model.calc(data, x)
                        model.calcDiff(data, x)
                        generated.calc(terminal_data, x)
                        generated.calcDiff(terminal_data, x)
                        for field, value in sentinels.items():
                            np.testing.assert_array_equal(
                                getattr(terminal_data, field), value
                            )
                        self.assert_fields_close(
                            self,
                            terminal_data,
                            data,
                            (
                                "g",
                                "h",
                                "Lx",
                                "Lp",
                                "Lxx",
                                "Lpp",
                                "Lpx",
                                "Gx",
                                "Gp",
                                "Hx",
                                "Hp",
                            ),
                            rtol,
                            atol,
                        )
            finally:
                os.chdir(old_cwd)

    def test_parameterized_integrator_codegen_normal_constructor(self):
        pin_model = pinocchio.Model()
        joint_id = pin_model.addJoint(
            0, pinocchio.JointModelRY(), pinocchio.SE3.Identity(), "joint1"
        )
        pin_model.appendBodyToJoint(
            joint_id,
            pinocchio.Inertia.FromBox(1.0, 0.2, 0.3, 0.4),
            pinocchio.SE3.Identity(),
        )
        pin_model.addFrame(
            pinocchio.Frame(
                "contact",
                joint_id,
                pinocchio.SE3(np.eye(3), np.array([1.0, 0.0, 0.0])),
                pinocchio.OP_FRAME,
            )
        )
        pin_model.lowerPositionLimit[:] = -1.0
        pin_model.upperPositionLimit[:] = 1.0
        state64 = crocoddyl.StateMultibody(pin_model)
        body_name = state64.pinocchio.names[1]

        def friction_actuation(module, dtype, state):
            joint_id = next(
                jid
                for jid in range(1, state.pinocchio.njoints)
                if state.pinocchio.joints[jid].nv == 1
            )
            friction = module.JointDynamicsModelFriction(
                joint_id,
                state.pinocchio.joints[joint_id].nq,
                np.log(np.array([0.15, 3.0, 0.2], dtype=dtype)),
                crocoddyl.JointFrictionType.COULOMB_VISCOUS,
            )
            return module.ActuationModelMultibody(state, [friction])

        def parameter_terms(module, dtype, state, nu, manager):
            parameter_residual = module.ResidualModelParameters(
                state, np.asarray(manager.zero(), dtype=dtype), nu
            )
            costs = module.CostModelSum(state, nu, manager.np)
            costs.addCost(
                "parameters",
                module.CostModelResidual(state, parameter_residual),
                0.7,
            )
            constraints = module.ConstraintModelManager(state, nu, manager.np)
            constraints.addConstraint(
                "a_inactive",
                module.ConstraintModelResidual(state, parameter_residual, True),
                False,
            )
            constraints.addConstraint(
                "b_inequality",
                module.ConstraintModelResidual(
                    state,
                    parameter_residual,
                    -np.ones(manager.np, dtype=dtype),
                    np.ones(manager.np, dtype=dtype),
                    True,
                ),
            )
            constraints.addConstraint(
                "c_equality",
                module.ConstraintModelResidual(state, parameter_residual, True),
            )
            state_residual = module.ResidualModelState(state, nu)
            constraints.addConstraint(
                "d_terminal_inequality",
                module.ConstraintModelResidual(
                    state,
                    state_residual,
                    -np.ones(state.ndx, dtype=dtype),
                    np.ones(state.ndx, dtype=dtype),
                    True,
                ),
            )
            constraints.addConstraint(
                "e_terminal_equality",
                module.ConstraintModelResidual(state, state_residual, True),
            )
            return costs, constraints

        def continuous_cases(module, dtype, state):
            actuation = friction_actuation(module, dtype, state)
            implicit = module.ImplicitConstraintModelMultiple(state, actuation.nu)
            dynamics = module.DynamicsModelConstrainedForward(
                state, actuation, implicit
            )
            integrator_time = module.IntegratorTime(0.012, True)
            manager = module.ParameterManager(state)
            manager.addParam("actuation", module.ActuationMultibodyParams(actuation))
            manager.addParam("a_inactive", module.LQRParams(state, 1), False)
            manager.addParam(
                "b_time", module.IntegratorTimeoptParams(state, integrator_time)
            )
            manager.addParam(
                "inertia",
                module.MultibodyInertialParams(
                    state,
                    module.LogCholeskyParametrization(),
                    [body_name],
                ),
            )
            costs, constraints = parameter_terms(
                module, dtype, state, actuation.nu, manager
            )
            models = [
                (
                    "euler",
                    module.IntegratedActionModelEuler(
                        dynamics, costs, constraints, None, integrator_time
                    ),
                )
            ]
            for name, scheme in (
                ("rk2", crocoddyl.RKType.two),
                ("rk3", crocoddyl.RKType.three),
                ("rk4", crocoddyl.RKType.four),
            ):
                models.append(
                    (
                        name,
                        module.IntegratedActionModelRK(
                            dynamics,
                            costs,
                            constraints,
                            None,
                            integrator_time,
                            scheme,
                        ),
                    )
                )
            p = np.asarray(manager.zero(), dtype=dtype)
            p += np.linspace(-0.03, 0.04, manager.np, dtype=dtype)
            p[0] = dtype(np.log(0.017))
            return [(name, model, manager, p) for name, model in models]

        def discretized_case(module, dtype, state):
            implicit = module.ImplicitConstraintModelMultiple(state, 0)
            frame_id = len(state.pinocchio.frames) - 1
            implicit.addConstraint(
                "contact",
                module.ContactModel(
                    state,
                    frame_id,
                    state.pinocchio.frames[frame_id].placement,
                    pinocchio.LOCAL_WORLD_ALIGNED,
                    0,
                    np.zeros(2, dtype=dtype),
                    [False, False, True, False, False, False],
                ),
            )
            dynamics = module.DynamicsModelImpulseForward(state, implicit)
            manager = module.ParameterManager(state)
            manager.addParam("a_inactive", module.LQRParams(state, 1), False)
            manager.addParam(
                "inertia",
                module.MultibodyInertialParams(
                    state,
                    module.LogCholeskyParametrization(),
                    [body_name],
                ),
            )
            costs, constraints = parameter_terms(module, dtype, state, 0, manager)
            model = module.DiscretizedActionModel(dynamics, costs, constraints)
            p = np.asarray(manager.zero(), dtype=dtype)
            p += np.linspace(-0.02, 0.03, manager.np, dtype=dtype)
            return "discretized", model, manager, p

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                scalar_cases = [
                    (
                        crocoddyl,
                        np.float64,
                        state64,
                        "float64",
                        1e-3,
                        2e-4,
                    )
                ]
                if PINOCCHIO_FLOAT32_AVAILABLE:
                    scalar_cases.append(
                        (
                            crocoddyl32,
                            np.float32,
                            state64.cast(crocoddyl.DType.Float32),
                            "float32",
                            6e-3,
                            6e-4,
                        )
                    )
                for module, dtype, state, suffix, rtol, atol in scalar_cases:
                    cases = continuous_cases(module, dtype, state)
                    cases.append(discretized_case(module, dtype, state))
                    for name, model, manager, p in cases:
                        with self.subTest(module=module.__name__, model=name):
                            bootstrap = model.createData(manager.createData())
                            model.set_params(bootstrap, manager)
                            model.update_p(bootstrap, p)
                            self.assertFalse(manager.getParamStatus("a_inactive"))

                            generated = module.ActionModelCodeGen(
                                model, "crocoddyl_" + name + "_" + suffix, True
                            )
                            direct = model.createData()
                            direct_terminal = model.createData()
                            codegen = generated.createData()
                            codegen_terminal = generated.createData()
                            generated.set_params(codegen, manager)
                            generated.set_params(codegen_terminal, manager)
                            model.update_p(direct, p)
                            model.update_p(direct_terminal, p)
                            generated.update_p(codegen, p)
                            generated.update_p(codegen_terminal, p)

                            dx = np.linspace(-0.06, 0.08, state.ndx, dtype=dtype)
                            x = np.asarray(
                                state.integrate(state.zero(), dx), dtype=dtype
                            )
                            u = np.linspace(-0.07, 0.09, model.nu, dtype=dtype)
                            model.calc(direct, x, u)
                            model.calcDiff(direct, x, u)
                            generated.calc(codegen, x, u)
                            generated.calcDiff(codegen, x, u)
                            self.assert_fields_close(
                                self,
                                codegen,
                                direct,
                                (
                                    "xnext",
                                    "g",
                                    "h",
                                    "Fx",
                                    "Fu",
                                    "Fp",
                                    "Lx",
                                    "Lu",
                                    "Lp",
                                    "Lxx",
                                    "Lxu",
                                    "Luu",
                                    "Lpp",
                                    "Lpx",
                                    "Lpu",
                                    "Gx",
                                    "Gu",
                                    "Gp",
                                    "Hx",
                                    "Hu",
                                    "Hp",
                                ),
                                rtol,
                                atol,
                            )
                            self.assertGreater(np.linalg.norm(codegen.Lp), 0.0)
                            self.assertGreater(np.linalg.norm(codegen.Fp), 0.0)
                            self.assertGreater(np.linalg.norm(codegen.Gp), 0.0)
                            self.assertGreater(np.linalg.norm(codegen.Hp), 0.0)

                            model.calc(direct_terminal, x)
                            model.calcDiff(direct_terminal, x)
                            generated.calc(codegen_terminal, x)
                            generated.calcDiff(codegen_terminal, x)
                            self.assert_fields_close(
                                self,
                                codegen_terminal,
                                direct_terminal,
                                (
                                    "xnext",
                                    "g",
                                    "h",
                                    "Fx",
                                    "Fp",
                                    "Lx",
                                    "Lp",
                                    "Lxx",
                                    "Lpp",
                                    "Lpx",
                                    "Gx",
                                    "Gp",
                                    "Hx",
                                    "Hp",
                                ),
                                rtol,
                                atol,
                            )
                            self.assertGreater(np.linalg.norm(codegen_terminal.Lp), 0.0)
                            self.assertGreater(codegen_terminal.Gp.shape[0], 0)
                            self.assertGreater(codegen_terminal.Hp.shape[0], 0)
                            self.assertFalse(manager.getParamStatus("a_inactive"))
            finally:
                os.chdir(old_cwd)

    def test_observer_codegen_bindings(self):
        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                pin_model = pinocchio.Model()
                joint_id = pin_model.addJoint(
                    0,
                    pinocchio.JointModelRY(),
                    pinocchio.SE3.Identity(),
                    "joint1",
                )
                pin_model.appendBodyToJoint(
                    joint_id,
                    pinocchio.Inertia.FromBox(1.0, 0.2, 0.3, 0.4),
                    pinocchio.SE3.Identity(),
                )
                pin_model.lowerPositionLimit[:] = -1.0
                pin_model.upperPositionLimit[:] = 1.0
                state64 = crocoddyl.StateMultibody(pin_model)
                scalar_cases = [
                    (
                        crocoddyl,
                        np.float64,
                        state64,
                        "float64",
                        1e-8,
                        1e-9,
                    )
                ]
                if PINOCCHIO_FLOAT32_AVAILABLE:
                    scalar_cases.append(
                        (
                            crocoddyl32,
                            np.float32,
                            state64.cast(crocoddyl.DType.Float32),
                            "float32",
                            3e-4,
                            3e-5,
                        )
                    )
                for module, dtype, state, suffix, rtol, atol in scalar_cases:
                    with self.subTest(module=module.__name__):
                        actuation = module.ActuationModelMultibody(state)
                        implicit = module.ImplicitConstraintModelMultiple(
                            state, actuation.nu
                        )
                        dynamics = module.DynamicsModelConstrainedForward(
                            state, actuation, implicit
                        )
                        manager = module.ParameterManager(state)
                        manager.addParam(
                            "inertia",
                            module.MultibodyInertialParams(
                                state,
                                module.LogCholeskyParametrization(),
                                [state.pinocchio.names[1]],
                            ),
                        )
                        observer_nu = state.ndx + dynamics.nu
                        costs = module.CostModelSum(state, observer_nu, manager.np)
                        costs.addCost(
                            "state",
                            module.CostModelResidual(
                                state,
                                module.ResidualModelState(state, observer_nu),
                            ),
                            0.4,
                        )
                        costs.addCost(
                            "parameters",
                            module.CostModelResidual(
                                state,
                                module.ResidualModelParameters(
                                    state,
                                    np.asarray(manager.zero(), dtype=dtype),
                                    observer_nu,
                                ),
                            ),
                            0.3,
                        )
                        constraints = module.ConstraintModelManager(
                            state, observer_nu, manager.np
                        )
                        constraints.addConstraint(
                            "control",
                            module.ConstraintModelResidual(
                                state,
                                module.ResidualModelControl(state, observer_nu),
                            ),
                        )
                        parameter_residual = module.ResidualModelParameters(
                            state,
                            np.asarray(manager.zero(), dtype=dtype),
                            observer_nu,
                        )
                        constraints.addConstraint(
                            "parameter_inequality",
                            module.ConstraintModelResidual(
                                state,
                                parameter_residual,
                                -np.ones(manager.np, dtype=dtype),
                                np.ones(manager.np, dtype=dtype),
                                True,
                            ),
                        )
                        constraints.addConstraint(
                            "parameter_equality",
                            module.ConstraintModelResidual(
                                state, parameter_residual, True
                            ),
                        )
                        state_residual = module.ResidualModelState(state, observer_nu)
                        constraints.addConstraint(
                            "state_inequality",
                            module.ConstraintModelResidual(
                                state,
                                state_residual,
                                -np.ones(state.ndx, dtype=dtype),
                                np.ones(state.ndx, dtype=dtype),
                                True,
                            ),
                        )
                        constraints.addConstraint(
                            "state_equality",
                            module.ConstraintModelResidual(state, state_residual, True),
                        )
                        observer = module.IntegratedObserverModelEuler(
                            dynamics, costs, constraints
                        )
                        bootstrap = observer.createData(manager.createData())
                        observer.set_params(bootstrap, manager)
                        p = np.asarray(manager.zero(), dtype=dtype)
                        p += np.linspace(-0.02, 0.03, manager.np, dtype=dtype)
                        data = observer.createData()
                        x = np.asarray(state.zero(), dtype=dtype)
                        w = np.linspace(-0.01, 0.01, observer.nu, dtype=dtype)
                        tau = np.linspace(-0.2, 0.2, observer.ntau, dtype=dtype)
                        observer.update_tau(tau)
                        observer.update_p(data, p)
                        observer.calc(data, x, w)
                        observer.calcDiff(data, x, w)

                        lib_name = "crocoddyl_observer_" + suffix
                        generated = module.ObserverModelCodeGen(
                            observer, lib_name, True
                        )
                        generated_data = generated.createData()
                        generated.set_params(generated_data, manager)
                        generated.update_tau(tau)
                        generated.update_p(generated_data, p)
                        generated.calc(generated_data, x, w)
                        generated.calcDiff(generated_data, x, w)
                        self.assertTrue(generated.existLib(lib_name))
                        self.assertIsInstance(
                            generated_data, module.ObserverDataCodeGen
                        )
                        self.assert_fields_close(
                            self,
                            generated_data,
                            data,
                            (
                                "xnext",
                                "g",
                                "h",
                                "Fx",
                                "Fu",
                                "Fp",
                                "Lx",
                                "Lu",
                                "Lp",
                                "Lxx",
                                "Lxu",
                                "Luu",
                                "Lpp",
                                "Lpx",
                                "Lpu",
                                "Gx",
                                "Gu",
                                "Gp",
                                "Hx",
                                "Hu",
                                "Hp",
                            ),
                            rtol,
                            atol,
                        )
                        self.assertGreater(np.linalg.norm(generated_data.Fp), 0.0)
                        self.assertAlmostEqual(
                            generated_data.cost,
                            data.cost,
                            delta=max(atol, abs(data.cost) * rtol),
                        )

                        loaded = module.ObserverModelCodeGen(lib_name, observer)
                        loaded_data = loaded.createData()
                        loaded.set_params(loaded_data, manager)
                        loaded.update_tau(tau)
                        loaded.update_p(loaded_data, p)
                        loaded.calc(loaded_data, x, w)
                        loaded.calcDiff(loaded_data, x, w)
                        self.assert_fields_close(
                            self,
                            loaded_data,
                            generated_data,
                            ("xnext", "Fx", "Fu", "Fp", "Gx", "Gu", "Gp"),
                            rtol,
                            atol,
                        )

                        observer.calc(data, x)
                        observer.calcDiff(data, x)
                        generated.calc(generated_data, x)
                        generated.calcDiff(generated_data, x)
                        self.assert_fields_close(
                            self,
                            generated_data,
                            data,
                            (
                                "xnext",
                                "g",
                                "h",
                                "Fx",
                                "Fp",
                                "Lx",
                                "Lp",
                                "Lxx",
                                "Lpp",
                                "Lpx",
                                "Gx",
                                "Gp",
                                "Hx",
                                "Hp",
                            ),
                            rtol,
                            atol,
                        )
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
