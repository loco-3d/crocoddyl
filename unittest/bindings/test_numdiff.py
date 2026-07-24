###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################

import copy
import gc
import unittest

import numpy as np
import pinocchio

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


def assert_close(test, actual, expected, dtype):
    tolerance = 2e-3 if dtype == np.float32 else 2e-7
    test.assertTrue(np.allclose(actual, expected, rtol=tolerance, atol=tolerance))


class DynamicsProbe:
    @staticmethod
    def make(module, dtype):
        class Model(module.DynamicsModelAbstract):
            def __init__(self):
                super().__init__(
                    module.StateVector(4),
                    crocoddyl.DynamicsType.ContinuousControl,
                    0,
                    2,
                    1,
                    1,
                )

            def calc(self, data, x, u=None):
                control = np.zeros(2, dtype=dtype) if u is None else u
                data.vdot = np.array(
                    [x[0] + 2 * x[2] + control[0], x[1] - x[3] + control[1]],
                    dtype=dtype,
                )
                data.dissipative_P = np.array([x[2] - 3 * x[3]], dtype=dtype)
                data.g = np.array([x[0] + control[0]], dtype=dtype)
                data.h = np.array([x[1] - control[1]], dtype=dtype)

            def calcDiff_xu(self, data, x, u=None):
                pass

        return Model()


class ActuationProbe:
    @staticmethod
    def make(module, dtype):
        class Model(module.ActuationModelAbstract):
            def __init__(self):
                super().__init__(module.StateVector(4), 2)

            def calc(self, data, x, u):
                data.tau = np.array([x[0] + 2 * u[0], x[1] - 3 * u[1]], dtype=dtype)

            def calcDiff(self, data, x, u):
                pass

            def commands(self, data, x, tau):
                data.u = np.array(
                    [(tau[0] - x[0]) / 2, (x[1] - tau[1]) / 3], dtype=dtype
                )

        return Model()


class NumDiffBindingsTest(unittest.TestCase):
    def check_action(self, module, dtype, cast_dtype):
        nx, nu, np_, ng, nh = 4, 2, 2, 1, 1
        model = module.ActionModelLQR(nx, nu, np_, ng, nh, False)
        model.A = np.eye(nx, dtype=dtype)
        model.B = np.arange(1, nx * nu + 1, dtype=dtype).reshape(nx, nu) / 10
        model.P = np.arange(1, nx * np_ + 1, dtype=dtype).reshape(nx, np_) / 7
        model.Q = np.eye(nx, dtype=dtype) * 2
        model.R = np.eye(nu, dtype=dtype) * 3
        model.N = np.arange(nx * nu, dtype=dtype).reshape(nx, nu) / 20
        model.W = np.eye(np_, dtype=dtype) * 4
        model.Y = np.arange(nx * np_, dtype=dtype).reshape(nx, np_) / 30
        model.V = np.arange(nu * np_, dtype=dtype).reshape(nu, np_) / 15
        model.G = np.arange(1, nx + nu + np_ + 1, dtype=dtype).reshape(1, -1)
        model.H = -np.asarray(model.G).reshape(1, -1)

        legacy_numdiff = module.ActionModelNumDiff(model, True)
        self.assertEqual(legacy_numdiff.np, 0)
        legacy_data = legacy_numdiff.createData()
        legacy_x = np.linspace(-0.4, 0.5, nx, dtype=dtype)
        legacy_u = np.linspace(0.2, 0.6, nu, dtype=dtype)
        legacy_numdiff.calc(legacy_data, legacy_x, legacy_u)
        legacy_numdiff.calcDiff(legacy_data, legacy_x, legacy_u)
        self.assertEqual(legacy_numdiff.cast(cast_dtype).np, 0)
        with self.assertRaises(Exception):
            module.ActionModelNumDiff(model, None, True)

        manager = module.ParameterManager(model.state)
        manager.addParam("lqr", module.LQRParams(model.state, np_))
        manager.addParam("inactive", module.LQRParams(model.state, 1), False)
        numdiff = module.ActionModelNumDiff(model, manager, False)
        manager_data = manager.createData()
        data = numdiff.createData(manager_data)
        self.assertIsInstance(data, module.ActionDataNumDiff)
        self.assertIs(data.params_data, manager_data)
        self.assertEqual(len(data.data_p), np_)

        x = np.linspace(-0.4, 0.5, nx, dtype=dtype)
        u = np.linspace(0.2, 0.6, nu, dtype=dtype)
        p = np.linspace(-0.3, 0.7, np_, dtype=dtype)
        numdiff.update_p(data, p)
        numdiff.calc(data, x, u)
        numdiff.calcDiff(data, x, u)

        exact = model.createData(manager_data)
        model.set_params(exact, manager)
        model.update_p(exact, p)
        model.calc(exact, x, u)
        model.calcDiff(exact, x, u)
        for name in (
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
        ):
            assert_close(self, getattr(data, name), getattr(exact, name), dtype)

        gauss_numdiff = module.ActionModelNumDiff(model, manager, True)
        gauss_data = gauss_numdiff.createData(manager.createData())
        gauss_numdiff.update_p(gauss_data, p)
        gauss_numdiff.calc(gauss_data, x, u)
        gauss_numdiff.calcDiff(gauss_data, x, u)
        for name in ("Lpp", "Lpx", "Lpu"):
            assert_close(self, getattr(gauss_data, name), getattr(exact, name), dtype)

        copied = copy.deepcopy(data)
        self.assertTrue(np.array_equal(copied.Fp, data.Fp))
        casted = numdiff.cast(cast_dtype)
        self.assertEqual((casted.state.nx, casted.nu, casted.np), (nx, nu, np_))
        with self.assertRaises(Exception):
            numdiff.update_p(data, np.zeros(np_ + 1, dtype=dtype))
        wrong_manager = module.ParameterManager(model.state)
        wrong_manager.addParam("wrong", module.LQRParams(model.state, np_ + 1))
        with self.assertRaises(Exception):
            numdiff.createData(wrong_manager.createData())
        with self.assertRaises(Exception):
            module.ActionModelNumDiff(None)
        with self.assertRaises(Exception):
            module.ActionDataNumDiff(None)

    def check_dynamics(self, module, dtype, cast_dtype):
        model = DynamicsProbe.make(module, dtype)
        with self.assertRaises(Exception):
            module.DynamicsModelNumDiff(model, None)
        numdiff = module.DynamicsModelNumDiff(model)
        data = numdiff.createData()
        self.assertIsInstance(data, module.DynamicsDataNumDiff)
        x = np.array([0.2, -0.4, 0.7, -0.3], dtype=dtype)
        u = np.array([0.6, -0.8], dtype=dtype)
        numdiff.calc(data, x, u)
        numdiff.calcDiff(data, x, u)
        assert_close(
            self,
            data.Fx,
            np.array([[1, 0, 2, 0], [0, 1, 0, -1]], dtype=dtype),
            dtype,
        )
        assert_close(self, data.Fu, np.eye(2, dtype=dtype), dtype)
        assert_close(self, data.dP_dv, np.array([[1, -3]], dtype=dtype), dtype)
        assert_close(self, data.Gx, np.array([[1, 0, 0, 0]], dtype=dtype), dtype)
        assert_close(self, data.Gu, np.array([[1, 0]], dtype=dtype), dtype)
        assert_close(self, data.Hx, np.array([[0, 1, 0, 0]], dtype=dtype), dtype)
        assert_close(self, data.Hu, np.array([[0, -1]], dtype=dtype), dtype)

        numdiff.calc(data, x)
        numdiff.calcDiff(data, x)
        self.assertEqual(data.Fu.shape, (2, 2))
        self.assertIsInstance(copy.copy(data), module.DynamicsDataNumDiff)
        self.assertEqual(numdiff.cast(cast_dtype).state.nx, 4)
        with self.assertRaises(Exception):
            numdiff.calcDiff(data, np.zeros(5, dtype=dtype), u)
        with self.assertRaises(Exception):
            module.DynamicsModelNumDiff(None)
        with self.assertRaises(Exception):
            module.DynamicsDataNumDiff(None)

        state64 = crocoddyl.StateMultibody(pinocchio.buildSampleModelHumanoidRandom())
        state = (
            state64 if module is crocoddyl else state64.cast(crocoddyl.DType.Float32)
        )

        class DiscreteModel(module.DynamicsModelAbstract):
            def __init__(self):
                super().__init__(state, crocoddyl.DynamicsType.DiscreteTime, 0, 2, 0, 0)
                self.B = np.column_stack(
                    (
                        np.linspace(-0.2, 0.3, state.ndx, dtype=dtype),
                        np.linspace(0.4, -0.1, state.ndx, dtype=dtype),
                    )
                )

            def calc(self, data, x, u=None):
                control = np.zeros(2, dtype=dtype) if u is None else u
                data.vdot = state.integrate(x, self.B @ control)

            def calcDiff_xu(self, data, x, u=None):
                pass

        self.assertNotEqual(state.nx, state.ndx)
        discrete = DiscreteModel()
        discrete_numdiff = module.DynamicsModelNumDiff(discrete)
        discrete_numdiff.disturbance = float(2e-3 if dtype == np.float32 else 1e-7)
        discrete_data = discrete_numdiff.createData()
        manifold_x = state.integrate(
            state.zero(), np.linspace(-0.1, 0.15, state.ndx, dtype=dtype)
        )
        manifold_u = np.zeros(2, dtype=dtype)
        discrete_numdiff.calc(discrete_data, manifold_x, manifold_u)
        discrete_numdiff.calcDiff(discrete_data, manifold_x, manifold_u)
        self.assertEqual(discrete_data.Fu.shape, (state.ndx, 2))
        assert_close(self, discrete_data.Fu, discrete.B, dtype)

    def check_actuation(self, module, dtype, cast_dtype):
        model = ActuationProbe.make(module, dtype)
        numdiff = module.ActuationModelNumDiff(model)
        data = numdiff.createData()
        self.assertIsInstance(data, module.ActuationDataNumDiff)
        x = np.array([0.4, -0.5, 0.2, -0.3], dtype=dtype)
        u = np.array([0.7, -0.2], dtype=dtype)
        numdiff.calc(data, x, u)
        numdiff.calcDiff(data, x, u)
        assert_close(self, data.dtau_dx, np.array([[1, 0, 0, 0], [0, 1, 0, 0]]), dtype)
        assert_close(self, data.dtau_du, np.diag([2, -3]).astype(dtype), dtype)
        numdiff.commands(data, x, data.tau)
        assert_close(self, data.u, u, dtype)
        self.assertIsInstance(copy.copy(data), module.ActuationDataNumDiff)
        self.assertEqual(numdiff.cast(cast_dtype).state.nx, 4)
        with self.assertRaises(Exception):
            numdiff.disturbance = -1
        with self.assertRaises(Exception):
            module.ActuationModelNumDiff(None)
        with self.assertRaises(Exception):
            module.ActuationDataNumDiff(None)

    def check_residual(self, module, dtype, cast_dtype):
        model = module.ResidualModelControl(module.StateVector(4), 2)
        numdiff = module.ResidualModelNumDiff(model)
        shared = module.DataCollectorAbstract()
        data = numdiff.createData(shared)
        self.assertIsInstance(data, module.ResidualDataNumDiff)
        x = np.array([0.3, -0.4, 0.8, 0.2], dtype=dtype)
        u = np.array([-0.2, 0.5], dtype=dtype)
        numdiff.calc(data, x, u)
        numdiff.calcDiff(data, x, u)
        assert_close(self, data.Rx, np.zeros((2, 4), dtype=dtype), dtype)
        assert_close(self, data.Ru, np.eye(2, dtype=dtype), dtype)
        numdiff.calc(data, x)
        numdiff.calcDiff(data, x)
        self.assertIsInstance(copy.copy(data), module.ResidualDataNumDiff)
        self.assertEqual(numdiff.cast(cast_dtype).state.nx, 4)
        with self.assertRaises(Exception):
            numdiff.calc(data, np.zeros(5, dtype=dtype), u)
        with self.assertRaises(Exception):
            module.ResidualModelNumDiff(None)
        with self.assertRaises(Exception):
            module.ResidualDataNumDiff(None, shared)

        state = module.StateVector(4)
        manager = module.ParameterManager(state)
        manager.addParam("parameters", module.LQRParams(state, 2))
        manager_data = manager.createData()
        parameter_model = module.ResidualModelParameters(
            state, np.zeros(2, dtype=dtype), 2
        )
        parameter_numdiff = module.ResidualModelNumDiff(parameter_model, manager)
        inferred = parameter_numdiff.createData(manager_data)
        second_inferred = parameter_numdiff.createData(manager_data)
        explicit = parameter_numdiff.createData(manager_data, manager_data)
        for index, parameter_data in enumerate((inferred, second_inferred, explicit)):
            marker = np.array([index + 1, -index - 2], dtype=dtype)
            parameter_data.parameter_data.params.p = marker
            assert_close(self, manager_data.params.p, marker, dtype)
        p = np.array([0.3, -0.4], dtype=dtype)
        manager.update(manager_data, p)
        parameter_numdiff.calc(inferred, x, u)
        parameter_numdiff.calcDiff(inferred, x, u)
        assert_close(self, inferred.Rp, np.eye(2, dtype=dtype), dtype)
        parameter_numdiff.update_p(second_inferred, -p)
        parameter_numdiff.calc(second_inferred, x, u)
        parameter_numdiff.calcDiff(second_inferred, x, u)
        assert_close(self, second_inferred.Rp, np.eye(2, dtype=dtype), dtype)
        parameter_numdiff.update_p(explicit, p)
        manager.update(manager_data, p)
        other_manager_data = manager.createData()
        with self.assertRaises(Exception):
            parameter_numdiff.createData(manager_data, other_manager_data)

        for explicit_data in (False, True):

            def create_inferred_data():
                local_manager_data = manager.createData()
                if explicit_data:
                    return parameter_numdiff.createData(
                        local_manager_data, local_manager_data
                    )
                return parameter_numdiff.createData(local_manager_data)

            lifetime_data = create_inferred_data()
            gc.collect()
            parameter_numdiff.update_p(lifetime_data, p)
            parameter_numdiff.calc(lifetime_data, x, u)
            parameter_numdiff.calcDiff(lifetime_data, x, u)
            assert_close(self, lifetime_data.Rp, np.eye(2, dtype=dtype), dtype)
            del lifetime_data
            gc.collect()

            def retain_parameter_data():
                local_manager_data = manager.createData()
                if explicit_data:
                    local_data = parameter_numdiff.createData(
                        local_manager_data, local_manager_data
                    )
                else:
                    local_data = parameter_numdiff.createData(local_manager_data)
                return local_data.parameter_data

            retained_parameter_data = retain_parameter_data()
            gc.collect()
            manager.update(retained_parameter_data, -p)
            assert_close(self, retained_parameter_data.params.p, -p, dtype)
            del retained_parameter_data
            gc.collect()

    def check_module(self, module, dtype, cast_dtype):
        self.check_action(module, dtype, cast_dtype)
        self.check_dynamics(module, dtype, cast_dtype)
        self.check_actuation(module, dtype, cast_dtype)
        self.check_residual(module, dtype, cast_dtype)

    def test_float64(self):
        self.check_module(crocoddyl, np.float64, crocoddyl.DType.Float32)

    def test_float32(self):
        self.check_module(crocoddyl_float32, np.float32, crocoddyl.DType.Float64)


if __name__ == "__main__":
    unittest.main()
