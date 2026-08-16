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

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


class InertialParameterizationBindingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state64 = crocoddyl.StateMultibody(pinocchio.buildSampleModelManipulator())
        cls.state32 = cls.state64.cast(crocoddyl.DType.Float32)

    def scalar_cases(self):
        return (
            (crocoddyl, np.float64, self.state64, crocoddyl.DType.Float32),
            (
                crocoddyl_float32,
                np.float32,
                self.state32,
                crocoddyl.DType.Float64,
            ),
        )

    @staticmethod
    def finite_difference(model, data, p, dtype):
        eps = dtype(2e-3 if dtype == np.float32 else 1e-7)
        jacobian = np.zeros((10, 10), dtype=dtype)
        for i in range(10):
            plus = p.copy()
            minus = p.copy()
            plus[i] += eps
            minus[i] -= eps
            psi_plus = np.empty(10, dtype=dtype)
            psi_minus = np.empty(10, dtype=dtype)
            model.fromParametrization(data, psi_plus, plus)
            model.fromParametrization(data, psi_minus, minus)
            jacobian[:, i] = (psi_plus - psi_minus) / (dtype(2) * eps)
        return jacobian

    def test_abstract_dispatch_fallback_and_copy(self):
        for module, dtype, _, _ in self.scalar_cases():
            with self.subTest(module=module.__name__):

                class Probe(module.InertialParametrizationAbstract):
                    def __init__(self, scalar_dtype=dtype):
                        super().__init__()
                        self.calls = 0
                        self.dtype = scalar_dtype

                    def fromParametrization(self, data, psi, p):
                        self.calls += 1
                        psi[:] = np.asarray(p, dtype=self.dtype) + self.dtype(1)

                    def toParametrization(self, p, psi):
                        self.calls += 1
                        p[:] = np.asarray(psi, dtype=self.dtype) - self.dtype(1)

                    def updateParametrizationDerivative(self, data, dpsi_dp, p, psi):
                        self.calls += 1
                        dpsi_dp[:, :] = np.eye(10, dtype=self.dtype)

                model = Probe()
                data = model.createData()
                self.assertIsInstance(data, module.InertialParametrizationDataAbstract)
                self.assertIsInstance(copy.copy(data), type(data))
                p = np.linspace(-0.2, 0.3, 10, dtype=dtype)
                psi = np.zeros(10, dtype=dtype)
                recovered = np.zeros(10, dtype=dtype)
                jacobian = np.zeros((10, 10), dtype=dtype)
                model.fromParametrization(data, psi, p)
                model.toParametrization(recovered, psi)
                model.updateParametrizationDerivative(data, jacobian, recovered, psi)
                np.testing.assert_allclose(
                    recovered,
                    p,
                    rtol=5e-6 if dtype == np.float32 else 1e-12,
                    atol=1e-7 if dtype == np.float32 else 0.0,
                )
                np.testing.assert_allclose(jacobian, np.eye(10, dtype=dtype))
                self.assertEqual(model.calls, 3)

                fallback = module.InertialParametrizationAbstract()
                self.assertIsInstance(
                    fallback.createData(),
                    module.InertialParametrizationDataAbstract,
                )
                with self.assertRaises(RuntimeError):
                    fallback.fromParametrization(data, psi, p)

    def test_conversions_derivatives_edges_casts_and_copies(self):
        p_log64 = np.array([0.2, -0.1, 0.15, -0.2, 0.1, -0.25, 0.3, 0.05, -0.08, 0.12])
        p_exp64 = np.array(
            [
                np.log(3.0),
                0.1,
                -0.2,
                0.3,
                0.25,
                -0.3,
                0.2,
                np.log(0.4),
                np.log(0.6),
                np.log(0.8),
            ]
        )
        edge64 = np.array(
            [
                np.log(1e-6),
                1e-8,
                -1e-8,
                2e-8,
                1e-9,
                -1e-9,
                2e-9,
                np.log(0.5),
                np.log(0.5),
                np.log(0.5),
            ]
        )
        for module, dtype, _, cast_dtype in self.scalar_cases():
            target_module = crocoddyl_float32 if dtype == np.float64 else crocoddyl
            for model, p64 in (
                (module.LogCholeskyParametrization(), p_log64),
                (module.ExpEigenValueParametrization(), p_exp64),
                (module.ExpEigenValueParametrization(), edge64),
            ):
                with self.subTest(module=module.__name__, model=type(model).__name__):
                    p = p64.astype(dtype)
                    data = model.createData()
                    psi = np.zeros(10, dtype=dtype)
                    recovered = np.zeros(10, dtype=dtype)
                    psi_roundtrip = np.zeros(10, dtype=dtype)
                    jacobian = np.zeros((10, 10), dtype=dtype, order="C")
                    model.fromParametrization(data, psi, p)
                    model.toParametrization(recovered, psi)
                    model.fromParametrization(data, psi_roundtrip, recovered)
                    model.updateParametrizationDerivative(data, jacobian, p, psi)
                    tolerance = 3e-3 if dtype == np.float32 else 2e-6
                    np.testing.assert_allclose(
                        psi_roundtrip, psi, rtol=tolerance, atol=tolerance
                    )
                    np.testing.assert_allclose(
                        jacobian,
                        self.finite_difference(model, data, p, dtype),
                        rtol=2e-2 if dtype == np.float32 else 2e-5,
                        atol=2e-3 if dtype == np.float32 else 2e-6,
                    )
                    inertia = pinocchio.Inertia.FromDynamicParameters(
                        np.asarray(psi, dtype=np.float64)
                    )
                    self.assertGreater(inertia.mass, 0.0)
                    self.assertTrue(np.all(np.linalg.eigvalsh(inertia.inertia) > 0.0))
                    self.assertIsInstance(copy.copy(model), type(model))
                    self.assertIsInstance(copy.deepcopy(data), type(data))
                    casted = model.cast(cast_dtype)
                    self.assertEqual(casted.np, 10)
                    self.assertIsInstance(
                        casted,
                        getattr(target_module, type(model).__name__),
                    )

            with self.assertRaises(crocoddyl.Exception):
                module.LogCholeskyParametrizationData(None)
            with self.assertRaises(crocoddyl.Exception):
                module.ExpEigenValueParametrizationData(None)

    def test_multibody_layout_resolution_update_copy_and_cast(self):
        for module, dtype, state, cast_dtype in self.scalar_cases():
            with self.subTest(module=module.__name__):
                target_module = crocoddyl_float32 if dtype == np.float64 else crocoddyl
                pin_model = self.state64.pinocchio
                joint_name = pin_model.names[1]
                frame_name = next(
                    frame.name
                    for frame in pin_model.frames
                    if frame.parentJoint == 2 and frame.name != pin_model.names[2]
                )
                parametrization = module.LogCholeskyParametrization()
                toggled = module.MultibodyInertialParams(
                    state, parametrization, [joint_name]
                )
                one_body = toggled.createData()
                toggled.changeBodyStatus(joint_name, False)
                self.assertEqual(toggled.np, 0)
                self.assertEqual(toggled.body_names, [])
                self.assertEqual(toggled.lb.shape, (0,))
                self.assertEqual(toggled.ub.shape, (0,))
                self.assertEqual(toggled.zero().shape, (0,))
                self.assertEqual(toggled.rand().shape, (0,))
                self.assertFalse(toggled.checkData(one_body))
                no_bodies = toggled.createData()
                self.assertEqual(len(no_bodies.psi), 0)
                self.assertEqual(len(no_bodies.dpsi_dp), 0)
                toggled.update(no_bodies, np.zeros(0, dtype=dtype))
                toggled.changeBodyStatus(joint_name, True)
                self.assertEqual(toggled.np, 10)
                self.assertEqual(toggled.body_names, [joint_name])
                self.assertFalse(toggled.checkData(no_bodies))
                max_value = np.finfo(dtype).max
                np.testing.assert_array_equal(
                    toggled.lb, np.full(10, -max_value, dtype=dtype)
                )
                np.testing.assert_array_equal(
                    toggled.ub, np.full(10, max_value, dtype=dtype)
                )
                self.assertTrue(toggled.checkData(toggled.createData()))

                model = module.MultibodyInertialParams(
                    state, parametrization, [joint_name, frame_name]
                )
                self.assertEqual(model.np, 20)
                self.assertEqual(model.body_names, [joint_name, pin_model.names[2]])
                np.testing.assert_array_equal(
                    model.lb, np.full(20, -max_value, dtype=dtype)
                )
                np.testing.assert_array_equal(
                    model.ub, np.full(20, max_value, dtype=dtype)
                )
                data = model.createData()
                self.assertIsInstance(data, module.MultibodyInertialParamsData)
                self.assertIsInstance(data, module.DynamicsParamsDataAbstract)
                self.assertIsInstance(data, module.ParamsDataAbstract)
                self.assertEqual((data.np_action, data.np_dynamics), (0, 20))
                self.assertEqual(len(data.psi), 2)
                self.assertEqual(len(data.dpsi_dp), 2)
                self.assertIsInstance(data.psi, module.StdVec_Vector10)
                self.assertIsInstance(data.dpsi_dp, module.StdVec_Matrix10)
                for psi, dpsi_dp in zip(data.psi, data.dpsi_dp):
                    self.assertEqual(psi.shape, (10,))
                    self.assertEqual(dpsi_dp.shape, (10, 10))
                self.assertIsInstance(
                    data.parametrization, module.LogCholeskyParametrizationData
                )
                with self.assertRaises(AttributeError):
                    data.parametrization = parametrization.createData()
                with self.assertRaises(AttributeError):
                    model.body_names = []
                with self.assertRaises(AttributeError):
                    model.parametrization = module.LogCholeskyParametrization()
                self.assertTrue(model.checkData(data))
                collector = module.DataCollectorParams(data)
                collector.params.p = np.ones(20, dtype=dtype)
                self.assertTrue(np.array_equal(data.p, np.ones(20, dtype=dtype)))
                p0 = np.asarray(model.zero(), dtype=dtype)
                before = None
                if dtype == np.float64:
                    before = [
                        np.asarray(inertia.toDynamicParameters()).copy()
                        for inertia in state.pinocchio.inertias
                    ]
                p = p0.copy()
                p[0] += dtype(0.05)
                p[11] -= dtype(0.03)
                model.update(data, p)
                self.assertTrue(np.array_equal(data.p, p))
                if before is not None:
                    self.assertFalse(
                        np.allclose(
                            state.pinocchio.inertias[1].toDynamicParameters(),
                            before[1],
                        )
                    )
                    self.assertFalse(
                        np.allclose(
                            state.pinocchio.inertias[2].toDynamicParameters(),
                            before[2],
                        )
                    )
                    for jid in range(3, pin_model.njoints):
                        np.testing.assert_allclose(
                            state.pinocchio.inertias[jid].toDynamicParameters(),
                            before[jid],
                        )
                np.testing.assert_allclose(model.zero(), p)

                copied_model = copy.copy(model)
                copied_data = copy.deepcopy(data)
                self.assertEqual(copied_model.body_names, model.body_names)
                self.assertTrue(np.array_equal(copied_data.p, data.p))
                self.assertIsNot(copied_data.parametrization, data.parametrization)
                copied_psi = module.StdVec_Vector10()
                copied_psi.append(np.full(10, dtype(2), dtype=dtype))
                copied_psi.append(np.asarray(copied_data.psi[1], dtype=dtype).copy())
                copied_data.psi = copied_psi
                copied_dpsi_dp = module.StdVec_Matrix10()
                copied_dpsi_dp.append(np.full((10, 10), dtype(3), dtype=dtype))
                copied_dpsi_dp.append(
                    np.asarray(copied_data.dpsi_dp[1], dtype=dtype).copy()
                )
                copied_data.dpsi_dp = copied_dpsi_dp
                self.assertFalse(np.array_equal(copied_data.psi[0], data.psi[0]))
                self.assertFalse(
                    np.array_equal(copied_data.dpsi_dp[0], data.dpsi_dp[0])
                )
                data.p = np.zeros(20, dtype=dtype)
                self.assertTrue(np.array_equal(copied_data.p, p))

                original_lb = np.linspace(-20, -1, 20, dtype=dtype)
                original_ub = np.linspace(1, 20, 20, dtype=dtype)
                model.lb = original_lb
                model.ub = original_ub
                model.changeBodyStatus(joint_name, True)
                model.changeBodyStatus("missing-body", True)
                model.changeBodyStatus("universe", True)
                self.assertEqual(model.np, 20)
                self.assertTrue(model.checkData(data))
                np.testing.assert_array_equal(model.lb, original_lb)
                np.testing.assert_array_equal(model.ub, original_ub)

                third_name = pin_model.names[3]
                model.changeBodyStatus(third_name, True)
                self.assertEqual(model.np, 30)
                self.assertEqual(
                    model.body_names,
                    [joint_name, pin_model.names[2], third_name],
                )
                np.testing.assert_array_equal(model.lb[:20], original_lb)
                np.testing.assert_array_equal(model.ub[:20], original_ub)
                np.testing.assert_array_equal(
                    model.lb[20:], np.full(10, -max_value, dtype=dtype)
                )
                np.testing.assert_array_equal(
                    model.ub[20:], np.full(10, max_value, dtype=dtype)
                )
                self.assertFalse(model.checkData(data))
                with self.assertRaises(crocoddyl.Exception):
                    model.update(data, model.zero())
                self.assertEqual(model.zero().shape, (30,))
                self.assertEqual(model.rand().shape, (30,))
                expanded = model.createData()
                self.assertEqual(len(expanded.psi), 3)

                model.changeBodyStatus(frame_name, False)
                self.assertEqual(model.np, 20)
                self.assertEqual(model.body_names, [joint_name, third_name])
                np.testing.assert_array_equal(model.lb[:10], original_lb[:10])
                np.testing.assert_array_equal(model.ub[:10], original_ub[:10])
                np.testing.assert_array_equal(
                    model.lb[10:], np.full(10, -max_value, dtype=dtype)
                )
                np.testing.assert_array_equal(
                    model.ub[10:], np.full(10, max_value, dtype=dtype)
                )
                self.assertFalse(model.checkData(expanded))
                reduced = model.createData()
                model.changeBodyStatus(frame_name, False)
                self.assertTrue(model.checkData(reduced))

                model.changeBodyStatus(frame_name, True)
                self.assertEqual(model.np, 30)
                self.assertEqual(
                    model.body_names,
                    [joint_name, third_name, pin_model.names[2]],
                )
                self.assertFalse(model.checkData(reduced))
                np.testing.assert_array_equal(model.lb[:10], original_lb[:10])
                np.testing.assert_array_equal(model.ub[:10], original_ub[:10])
                np.testing.assert_array_equal(
                    model.lb[10:], np.full(20, -max_value, dtype=dtype)
                )
                np.testing.assert_array_equal(
                    model.ub[10:], np.full(20, max_value, dtype=dtype)
                )
                final_data = model.createData()
                model.update(final_data, model.zero())
                self.assertTrue(
                    all(np.linalg.norm(jacobian) > 0 for jacobian in final_data.dpsi_dp)
                )

                wrong_psi = model.createData()
                short_psi = module.StdVec_Vector10()
                short_psi.append(np.zeros(10, dtype=dtype))
                wrong_psi.psi = short_psi
                self.assertFalse(model.checkData(wrong_psi))
                with self.assertRaises(crocoddyl.Exception):
                    model.update(wrong_psi, model.zero())
                wrong_dpsi_dp = model.createData()
                short_dpsi_dp = module.StdVec_Matrix10()
                short_dpsi_dp.append(np.zeros((10, 10), dtype=dtype))
                wrong_dpsi_dp.dpsi_dp = short_dpsi_dp
                self.assertFalse(model.checkData(wrong_dpsi_dp))
                with self.assertRaises(crocoddyl.Exception):
                    model.update(wrong_dpsi_dp, model.zero())

                independent_model = copy.copy(model)
                independent_model.changeBodyStatus(joint_name, False)
                self.assertEqual(independent_model.np, 20)
                self.assertEqual(model.np, 30)
                casted = model.cast(cast_dtype)
                self.assertIsInstance(casted, target_module.MultibodyInertialParams)
                self.assertEqual(casted.np, 30)
                self.assertEqual(casted.body_names, model.body_names)
                expected_lb = np.empty_like(casted.lb)
                expected_ub = np.empty_like(casted.ub)
                cast_max = np.finfo(casted.lb.dtype).max
                lb_unbounded = model.lb == -max_value
                ub_unbounded = model.ub == max_value
                expected_lb[lb_unbounded] = -cast_max
                expected_ub[ub_unbounded] = cast_max
                expected_lb[~lb_unbounded] = model.lb[~lb_unbounded]
                expected_ub[~ub_unbounded] = model.ub[~ub_unbounded]
                np.testing.assert_allclose(casted.lb, expected_lb, rtol=0, atol=0)
                np.testing.assert_allclose(casted.ub, expected_ub, rtol=0, atol=0)

                default_model = module.MultibodyInertialParams(
                    state, module.ExpEigenValueParametrization()
                )
                self.assertEqual(default_model.np, 10 * (pin_model.njoints - 1))
                empty = module.MultibodyInertialParams(
                    state, module.LogCholeskyParametrization(), []
                )
                self.assertEqual(empty.np, 0)
                with self.assertRaises(crocoddyl.Exception):
                    module.MultibodyInertialParams(
                        state, parametrization, [joint_name, joint_name]
                    )
                with self.assertRaises(crocoddyl.Exception):
                    module.MultibodyInertialParams(state, parametrization, ["universe"])
                with self.assertRaises(crocoddyl.Exception):
                    module.MultibodyInertialParams(
                        state, parametrization, ["missing-body"]
                    )
                with self.assertRaises(crocoddyl.Exception):
                    module.MultibodyInertialParamsData(None)

    def test_parameter_manager_and_regressor_binding(self):
        for module, dtype, state, _ in self.scalar_cases():
            with self.subTest(module=module.__name__):
                body_name = self.state64.pinocchio.names[1]
                inertial = module.MultibodyInertialParams(
                    state,
                    module.ExpEigenValueParametrization(),
                    [body_name],
                )
                manager = module.ParameterManager(state)
                manager.addParam("inertial", inertial)
                manager_data = manager.createData()
                self.assertIsInstance(
                    manager_data.dynamics_params["inertial"],
                    module.MultibodyInertialParamsData,
                )
                p = np.asarray(manager.zero(), dtype=dtype)
                p[0] += dtype(0.05)
                manager.update(manager_data, p)
                self.assertTrue(np.array_equal(manager_data.params.p, p))

                actuation = module.ActuationModelMultibody(state)
                constraints = module.ImplicitConstraintModelMultiple(
                    state, actuation.nu
                )
                dynamics = module.DynamicsModelConstrainedForward(
                    state, actuation, constraints
                )
                data = dynamics.createData(manager_data)
                dynamics.set_params(data, manager)
                dynamics.update_p(data, p)
                x = np.asarray(state.rand(), dtype=dtype)
                u = np.linspace(-0.1, 0.2, dynamics.nu, dtype=dtype)
                dynamics.calc(data, x, u)
                dynamics.calcDiff(data, x, u)
                self.assertEqual(data.Fp.shape, (state.nv, 10))
                self.assertTrue(np.all(np.isfinite(data.Fp)))
                manager_data.params.p = p + dtype(0.01)
                self.assertTrue(
                    np.array_equal(data.multibody.params.p, p + dtype(0.01))
                )

                second_body = self.state64.pinocchio.names[2]
                inertial.changeBodyStatus(second_body, True)
                self.assertEqual(inertial.np, 20)
                with self.assertRaises(crocoddyl.Exception):
                    manager.update(manager_data, p)
                with self.assertRaises(crocoddyl.Exception):
                    dynamics.update_p(data, p)

                reconfigured = module.MultibodyInertialParams(
                    state,
                    module.ExpEigenValueParametrization(),
                    [body_name],
                )
                rebuilt = module.ParameterManager(state)
                rebuilt.addParam("inertial", reconfigured)
                old_data = rebuilt.createData()
                rebuilt.removeParam("inertial")
                reconfigured.changeBodyStatus(second_body, True)
                rebuilt.addParam("inertial", reconfigured)
                self.assertEqual(rebuilt.np, 20)
                with self.assertRaises(crocoddyl.Exception):
                    rebuilt.update(old_data, np.zeros(20, dtype=dtype))
                rebuilt_data = rebuilt.createData()
                rebuilt_p = np.zeros(20, dtype=dtype)
                rebuilt.update(rebuilt_data, rebuilt_p)
                self.assertTrue(np.array_equal(rebuilt_data.params.p, rebuilt_p))


if __name__ == "__main__":
    unittest.main()
