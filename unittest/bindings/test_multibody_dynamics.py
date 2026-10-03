###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################

import copy
import unittest

import numpy as np
import pinocchio

try:
    import pinocchio.float32

    PINOCCHIO_FLOAT32_AVAILABLE = True
except ModuleNotFoundError:
    PINOCCHIO_FLOAT32_AVAILABLE = False

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


class MultibodyDynamicsBindingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state64 = crocoddyl.StateMultibody(pinocchio.buildSampleModelManipulator())
        cls.state32 = (
            cls.state64.cast(crocoddyl.DType.Float32)
            if PINOCCHIO_FLOAT32_AVAILABLE
            else None
        )

    def make_contact(self, module, state, dtype, nu, mask):
        frame_id = len(state.pinocchio.frames) - 1
        return module.ContactModel(
            state,
            frame_id,
            state.pinocchio.frames[frame_id].placement,
            pinocchio.LOCAL_WORLD_ALIGNED,
            nu,
            np.array([0.1, 0.2], dtype=dtype),
            mask,
        )

    def make_friction_actuation(self, module, state, dtype):
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

    def check_scalar(self, module, dtype, state, other_module, cast_dtype):
        tolerance = 2e-4 if dtype == np.float32 else 1e-10
        actuation = module.ActuationModelMultibody(state)
        constraints = module.ImplicitConstraintModelMultiple(state, actuation.nu)
        constraints.addConstraint(
            "contact",
            self.make_contact(
                module,
                state,
                dtype,
                actuation.nu,
                [True, True, True, False, False, False],
            ),
        )
        constraints.addConstraint(
            "inactive",
            self.make_contact(
                module,
                state,
                dtype,
                actuation.nu,
                [True, False, False, True, False, False],
            ),
            False,
        )
        forward = module.DynamicsModelConstrainedForward(state, actuation, constraints)
        forward_data = forward.createData()
        external_manager = module.ParameterManager(state)
        external_payload = external_manager.createData()
        external_forward_data = forward.createData(external_payload)
        self.assertIsInstance(forward_data, module.DynamicsDataConstrainedForward)
        self.assertIsInstance(
            external_forward_data, module.DynamicsDataConstrainedForward
        )
        self.assertEqual(external_forward_data.multibody.params.p.size, 0)
        self.assertIsInstance(forward_data, module.DynamicsDataAbstract)
        self.assertEqual((constraints.nc, constraints.nc_total), (3, 5))
        self.assertIs(forward.actuation, actuation)
        self.assertIs(forward.constraints, constraints)
        self.assertIsNone(forward.params)
        self.assertIsNotNone(forward_data.shared)
        with self.assertRaises(AttributeError):
            forward_data.shared = None

        x = np.asarray(state.rand(), dtype=dtype)
        u = np.linspace(-0.2, 0.3, forward.nu, dtype=dtype)
        forward.calc(forward_data, x, u)
        forward.calcDiff_xu(forward_data, x, u)
        forward.calcDiff_xu(forward_data, x)
        forward.calcDiff(forward_data, x, u)
        self.assertTrue(np.all(np.isfinite(forward_data.vdot)))
        self.assertTrue(np.all(np.isfinite(forward_data.Fx)))
        self.assertEqual(forward_data.Fu.shape, (state.nv, forward.nu))

        copied_forward = copy.copy(forward)
        copied_forward_data = copy.copy(forward_data)
        self.assertIsInstance(copied_forward, module.DynamicsModelConstrainedForward)
        self.assertTrue(np.array_equal(copied_forward_data.Fx, forward_data.Fx))
        forward_data.Fx = np.zeros_like(forward_data.Fx)
        self.assertFalse(np.array_equal(copied_forward_data.Fx, forward_data.Fx))
        if PINOCCHIO_FLOAT32_AVAILABLE:
            casted_forward = forward.cast(cast_dtype)
            self.assertIsInstance(
                casted_forward, other_module.DynamicsModelConstrainedForward
            )
            self.assertEqual(casted_forward.nu, forward.nu)

        forward_data.Fu = np.full(forward_data.Fu.shape, 4.0, dtype=dtype)
        forward.calc(forward_data, x)
        forward.calcDiff(forward_data, x)
        self.assertTrue(np.all(forward_data.Fu == dtype(4.0)))

        estimation = module.DynamicsModelConstrainedForward(
            state,
            actuation,
            constraints,
            0,
            crocoddyl.DynamicsType.ContinuousEstimation,
        )
        estimation_data = estimation.createData()
        estimation.update_tau(np.linspace(-0.1, 0.2, actuation.nu, dtype=dtype))
        estimation.calc(estimation_data, x, np.empty(0, dtype=dtype))
        estimation.calcDiff(estimation_data, x, np.empty(0, dtype=dtype))
        self.assertEqual(estimation.nu, 0)
        self.assertEqual(estimation_data.Fu.shape[1], 0)

        inverse_nu = state.nv + 3
        inverse_constraints = module.ImplicitConstraintModelMultiple(state, inverse_nu)
        inverse_constraints.addConstraint(
            "contact",
            self.make_contact(
                module,
                state,
                dtype,
                inverse_nu,
                [True, True, True, False, False, False],
            ),
        )
        inverse = module.DynamicsModelConstrainedInverse(
            state, actuation, inverse_constraints
        )
        inverse_data = inverse.createData()
        external_inverse_data = inverse.createData(external_payload)
        self.assertIsInstance(inverse_data, module.DynamicsDataConstrainedInverse)
        self.assertIsInstance(
            external_inverse_data, module.DynamicsDataConstrainedInverse
        )
        self.assertEqual(external_inverse_data.multibody.params.p.size, 0)
        inverse_u = np.linspace(-0.2, 0.2, inverse.nu, dtype=dtype)
        inverse.calc(inverse_data, x, inverse_u)
        inverse.calcDiff_xu(inverse_data, x, inverse_u)
        inverse.calcDiff_xu(inverse_data, x)
        inverse.calcDiff(inverse_data, x, inverse_u)
        self.assertEqual(inverse.nh, 3)
        self.assertEqual(inverse_data.h.shape, (3,))
        self.assertTrue(np.all(np.isfinite(inverse_data.Hx)))
        copied_inverse_data = copy.deepcopy(inverse_data)
        self.assertTrue(np.array_equal(copied_inverse_data.Hu, inverse_data.Hu))

        impulse_constraints = module.ImplicitConstraintModelMultiple(state, 0)
        impulse_constraints.addConstraint(
            "contact",
            self.make_contact(
                module,
                state,
                dtype,
                0,
                [True, True, True, False, False, False],
            ),
        )
        impulse = module.DynamicsModelImpulseForward(
            state, impulse_constraints, 0, 0.2, 0.1
        )
        impulse_data = impulse.createData()
        external_impulse_data = impulse.createData(external_payload)
        self.assertIsInstance(impulse_data, module.DynamicsDataImpulseForward)
        self.assertIsInstance(external_impulse_data, module.DynamicsDataImpulseForward)
        self.assertEqual(external_impulse_data.multibody.params.p.size, 0)
        empty_u = np.empty(0, dtype=dtype)
        impulse.calc(impulse_data, x, empty_u)
        impulse.calcDiff(impulse_data, x, empty_u)
        self.assertEqual(impulse.nu, 0)
        self.assertAlmostEqual(impulse.r_coeff, 0.2, delta=tolerance)
        self.assertAlmostEqual(impulse.JMinvJt_damping, 0.1, delta=tolerance)
        self.assertTrue(np.all(np.isfinite(impulse_data.vdot)))
        copied_impulse_data = copy.copy(impulse_data)
        self.assertTrue(np.array_equal(copied_impulse_data.Fx, impulse_data.Fx))
        impulse.calc(impulse_data, x)
        impulse.calcDiff(impulse_data, x)
        self.assertTrue(np.all(impulse_data.vdot == dtype(0.0)))
        self.assertTrue(np.all(impulse_data.Fx == dtype(0.0)))

        friction_actuation = self.make_friction_actuation(module, state, dtype)
        parameter_constraints = module.ImplicitConstraintModelMultiple(
            state, friction_actuation.nu
        )
        parameter_forward = module.DynamicsModelConstrainedForward(
            state, friction_actuation, parameter_constraints
        )
        parameter_data = parameter_forward.createData()
        manager = module.ParameterManager(state)
        manager.addParam(
            "actuation", module.ActuationMultibodyParams(friction_actuation)
        )
        parameter_forward.set_params(parameter_data, manager)
        p = np.asarray(manager.zero(), dtype=dtype) + dtype(0.05)
        parameter_forward.update_p(parameter_data, p)
        parameter_u = np.full(parameter_forward.nu, 0.3, dtype=dtype)
        parameter_forward.calc(parameter_data, x, parameter_u)
        parameter_forward.calcDiff(parameter_data, x, parameter_u)
        self.assertEqual(parameter_forward.np, manager.np)
        self.assertEqual(parameter_data.Fp.shape, (state.nv, manager.np))
        self.assertTrue(np.all(np.isfinite(parameter_data.Fp)))
        self.assertTrue(np.array_equal(parameter_data.multibody.params.p, p))
        shared_forward_payload = manager.createData()
        shared_forward_data = parameter_forward.createData(shared_forward_payload)
        parameter_forward.set_params(shared_forward_data, manager)
        parameter_forward.update_p(shared_forward_data, p)
        self.assertTrue(
            np.array_equal(
                shared_forward_payload.params.p, shared_forward_data.multibody.params.p
            )
        )
        shared_forward_payload.params.p = p + dtype(0.01)
        self.assertTrue(
            np.array_equal(shared_forward_data.multibody.params.p, p + dtype(0.01))
        )

        eps = dtype(2e-3 if dtype == np.float32 else 1e-6)
        derivative_tolerance = 2e-2 if dtype == np.float32 else 8e-4
        parameter_estimation_constraints = module.ImplicitConstraintModelMultiple(
            state, friction_actuation.nu
        )
        parameter_estimation = module.DynamicsModelConstrainedForward(
            state,
            friction_actuation,
            parameter_estimation_constraints,
            0,
            crocoddyl.DynamicsType.ContinuousEstimation,
        )
        parameter_estimation_data = parameter_estimation.createData()
        parameter_estimation.set_params(parameter_estimation_data, manager)
        parameter_estimation.update_tau(
            np.linspace(-0.15, 0.25, friction_actuation.nu, dtype=dtype)
        )
        parameter_estimation.update_p(parameter_estimation_data, p)
        empty_u = np.empty(0, dtype=dtype)
        parameter_estimation.calc(parameter_estimation_data, x, empty_u)
        parameter_estimation.calcDiff(parameter_estimation_data, x, empty_u)
        self.assertFalse(np.all(parameter_estimation_data.Fp == dtype(0)))

        for parameter_model, current_data, current_u in (
            (parameter_forward, parameter_data, parameter_u),
            (parameter_estimation, parameter_estimation_data, empty_u),
        ):
            finite_fp = np.zeros_like(current_data.Fp)
            plus = parameter_model.createData()
            minus = parameter_model.createData()
            for i in range(p.size):
                pp = p.copy()
                pm = p.copy()
                pp[i] += eps
                pm[i] -= eps
                parameter_model.update_p(plus, pp)
                parameter_model.calc(plus, x, current_u)
                parameter_model.update_p(minus, pm)
                parameter_model.calc(minus, x, current_u)
                finite_fp[:, i] = (plus.vdot - minus.vdot) / (dtype(2) * eps)
            parameter_model.update_p(current_data, p)
            parameter_model.calc(current_data, x, current_u)
            parameter_model.calcDiff(current_data, x, current_u)
            np.testing.assert_allclose(
                current_data.Fp,
                finite_fp,
                rtol=derivative_tolerance,
                atol=derivative_tolerance,
            )

        parameter_inverse_models = []
        for mode in (
            crocoddyl.DynamicsType.ContinuousControl,
            crocoddyl.DynamicsType.ContinuousEstimation,
        ):
            parameter_inverse_constraints = module.ImplicitConstraintModelMultiple(
                state, state.nv
            )
            parameter_inverse = module.DynamicsModelConstrainedInverse(
                state, friction_actuation, parameter_inverse_constraints, 0, mode
            )
            parameter_inverse_data = parameter_inverse.createData()
            parameter_inverse.set_params(parameter_inverse_data, manager)
            parameter_inverse.update_p(parameter_inverse_data, p)
            if mode == crocoddyl.DynamicsType.ContinuousEstimation:
                parameter_inverse.update_tau(
                    np.linspace(-0.1, 0.2, friction_actuation.nu, dtype=dtype)
                )
            parameter_inverse_u = np.linspace(
                -0.05, 0.08, parameter_inverse.nu, dtype=dtype
            )
            parameter_inverse.calc(parameter_inverse_data, x, parameter_inverse_u)
            parameter_inverse.calcDiff(parameter_inverse_data, x, parameter_inverse_u)
            if mode == crocoddyl.DynamicsType.ContinuousEstimation:
                self.assertFalse(np.all(parameter_inverse_data.Hp == dtype(0)))
            finite_hp = np.zeros_like(parameter_inverse_data.Hp)
            plus = parameter_inverse.createData()
            minus = parameter_inverse.createData()
            for i in range(p.size):
                pp = p.copy()
                pm = p.copy()
                pp[i] += eps
                pm[i] -= eps
                parameter_inverse.update_p(plus, pp)
                parameter_inverse.calc(plus, x, parameter_inverse_u)
                parameter_inverse.update_p(minus, pm)
                parameter_inverse.calc(minus, x, parameter_inverse_u)
                finite_hp[:, i] = (plus.h - minus.h) / (dtype(2) * eps)
            parameter_inverse.update_p(parameter_inverse_data, p)
            parameter_inverse.calc(parameter_inverse_data, x, parameter_inverse_u)
            parameter_inverse.calcDiff(parameter_inverse_data, x, parameter_inverse_u)
            np.testing.assert_allclose(
                parameter_inverse_data.Hp,
                finite_hp,
                rtol=derivative_tolerance,
                atol=derivative_tolerance,
            )
            parameter_inverse_models.append(parameter_inverse)
            shared_inverse_payload = manager.createData()
            shared_inverse_data = parameter_inverse.createData(shared_inverse_payload)
            parameter_inverse.set_params(shared_inverse_data, manager)
            parameter_inverse.update_p(shared_inverse_data, p)
            self.assertTrue(
                np.array_equal(
                    shared_inverse_payload.params.p,
                    shared_inverse_data.multibody.params.p,
                )
            )
            shared_inverse_payload.params.p = p + dtype(0.01)
            self.assertTrue(
                np.array_equal(shared_inverse_data.multibody.params.p, p + dtype(0.01))
            )

        parameter_impulse_constraints = module.ImplicitConstraintModelMultiple(state, 0)
        parameter_impulse_constraints.addConstraint(
            "contact",
            self.make_contact(
                module,
                state,
                dtype,
                0,
                [True, True, True, False, False, False],
            ),
        )
        parameter_impulse = module.DynamicsModelImpulseForward(
            state, parameter_impulse_constraints
        )
        parameter_impulse_data = parameter_impulse.createData()
        parameter_impulse.set_params(parameter_impulse_data, manager)
        parameter_impulse.update_p(parameter_impulse_data, p)
        parameter_impulse.calc(parameter_impulse_data, x, empty_u)
        parameter_impulse.calcDiff(parameter_impulse_data, x, empty_u)
        shared_impulse_payload = manager.createData()
        shared_impulse_data = parameter_impulse.createData(shared_impulse_payload)
        parameter_impulse.set_params(shared_impulse_data, manager)
        parameter_impulse.update_p(shared_impulse_data, p)
        self.assertTrue(
            np.array_equal(
                shared_impulse_payload.params.p,
                shared_impulse_data.multibody.params.p,
            )
        )
        shared_impulse_payload.params.p = p + dtype(0.01)
        self.assertTrue(
            np.array_equal(shared_impulse_data.multibody.params.p, p + dtype(0.01))
        )
        finite_impulse_fp = np.zeros_like(parameter_impulse_data.Fp)
        plus = parameter_impulse.createData()
        minus = parameter_impulse.createData()
        for i in range(p.size):
            pp = p.copy()
            pm = p.copy()
            pp[i] += eps
            pm[i] -= eps
            parameter_impulse.update_p(plus, pp)
            parameter_impulse.calc(plus, x, empty_u)
            parameter_impulse.update_p(minus, pm)
            parameter_impulse.calc(minus, x, empty_u)
            finite_impulse_fp[:, i] = (plus.vdot - minus.vdot) / (dtype(2) * eps)
        parameter_impulse.update_p(parameter_impulse_data, p)
        parameter_impulse.calc(parameter_impulse_data, x, empty_u)
        parameter_impulse.calcDiff(parameter_impulse_data, x, empty_u)
        np.testing.assert_allclose(
            parameter_impulse_data.Fp,
            finite_impulse_fp,
            rtol=derivative_tolerance,
            atol=derivative_tolerance,
        )

        if PINOCCHIO_FLOAT32_AVAILABLE:
            self.assertEqual(parameter_forward.cast(cast_dtype).np, manager.np)
            self.assertEqual(
                parameter_inverse_models[0].cast(cast_dtype).np, manager.np
            )
            self.assertEqual(parameter_impulse.cast(cast_dtype).np, manager.np)

        with self.assertRaises(crocoddyl.Exception):
            module.DynamicsModelConstrainedForward(None, actuation, constraints)
        with self.assertRaises(crocoddyl.Exception):
            module.DynamicsModelConstrainedForward(state, None, constraints)
        with self.assertRaises(crocoddyl.Exception):
            module.DynamicsModelConstrainedForward(state, actuation, None)
        with self.assertRaises(crocoddyl.Exception):
            module.DynamicsDataConstrainedForward(None)
        with self.assertRaises(crocoddyl.Exception):
            module.DynamicsDataConstrainedInverse(None)
        with self.assertRaises(crocoddyl.Exception):
            module.DynamicsDataImpulseForward(None)
        with self.assertRaises(crocoddyl.Exception):
            forward.calc(forward_data, np.zeros(state.nx + 1, dtype=dtype), u)
        with self.assertRaises(crocoddyl.Exception):
            forward.calc(forward_data, x, np.zeros(forward.nu + 1, dtype=dtype))
        with self.assertRaises(crocoddyl.Exception):
            impulse.calc(impulse_data, x, np.zeros(1, dtype=dtype))

    def test_float64(self):
        self.check_scalar(
            crocoddyl,
            np.float64,
            self.state64,
            crocoddyl_float32,
            crocoddyl.DType.Float32,
        )

    @unittest.skipUnless(
        PINOCCHIO_FLOAT32_AVAILABLE, "pinocchio.float32 is not available"
    )
    def test_float32(self):
        self.check_scalar(
            crocoddyl_float32,
            np.float32,
            self.state32,
            crocoddyl,
            crocoddyl.DType.Float64,
        )


if __name__ == "__main__":
    unittest.main()
