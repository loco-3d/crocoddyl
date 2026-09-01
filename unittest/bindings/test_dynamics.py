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

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


def make_probe(module, dtype, dynamics_type):
    class DynamicsProbe(module.DynamicsModelAbstract):
        def __init__(self):
            super().__init__(module.StateVector(4), dynamics_type, 3, 2, 2, 1)
            self.calc_calls = 0
            self.xu_calls = 0
            self.p_calls = 0
            self.create_calls = 0
            self.update_calls = 0
            self.last_p = None
            self.calc_terminal = []
            self.xu_terminal = []

        def calc(self, data, x, u=None):
            value = x.sum() + (0.0 if u is None else u.sum())
            data.vdot = np.full(data.vdot.shape, value, dtype=dtype)
            power = 0.0 if u is None else u @ u
            data.dissipative_P = np.array([power], dtype=dtype)
            data.h = np.full(1, value, dtype=dtype)
            data.g = np.full(2, -value, dtype=dtype)
            self.calc_calls += 1
            self.calc_terminal.append(u is None)

        def calcDiff_xu(self, data, x, u=None):
            data.Fx = np.ones(data.Fx.shape, dtype=dtype)
            data.Fu = np.eye(data.Fu.shape[0], data.Fu.shape[1], dtype=dtype)
            data.dP_dv = np.full(data.dP_dv.shape, 2.0, dtype=dtype)
            data.Hx = np.full(data.Hx.shape, 3.0, dtype=dtype)
            data.Hu = np.full(data.Hu.shape, 4.0, dtype=dtype)
            data.Gx = np.full(data.Gx.shape, 5.0, dtype=dtype)
            data.Gu = np.full(data.Gu.shape, 6.0, dtype=dtype)
            self.xu_calls += 1
            self.xu_terminal.append(u is None)

        def calcDiff_p(self, data, x, u):
            data.Fp = np.full(data.Fp.shape, 7.0, dtype=dtype)
            data.dP_dp = np.full(data.dP_dp.shape, 8.0, dtype=dtype)
            data.Hp = np.full(data.Hp.shape, 9.0, dtype=dtype)
            data.Gp = np.full(data.Gp.shape, 10.0, dtype=dtype)
            self.p_calls += 1

        def update_p(self, data, p):
            self.last_p = np.array(p, copy=True)
            self.update_calls += 1

        def createData(self):
            self.create_calls += 1
            data = module.DynamicsDataAbstract(self)
            data.vdot = np.ones(data.vdot.shape, dtype=dtype)
            return data

    return DynamicsProbe()


class DynamicsTest(unittest.TestCase):
    def check_model_data_and_dispatch(self, module, dtype, dynamics_type):
        model = make_probe(module, dtype, dynamics_type.ContinuousControl)
        self.assertEqual(model.state.nx, 4)
        self.assertEqual((model.np, model.nu, model.ng, model.nh), (3, 2, 2, 1))
        self.assertEqual(model.dyn_type, dynamics_type.ContinuousControl)
        self.assertTrue(np.array_equal(model.tau_meas, np.zeros(2, dtype=dtype)))
        self.assertTrue(np.all(np.isneginf(model.p_lb)))
        self.assertTrue(np.all(np.isposinf(model.p_ub)))

        model.p_lb = np.array([-3.0, -2.0, -1.0], dtype=dtype)
        model.p_ub = np.array([1.0, 2.0, 3.0], dtype=dtype)
        model.update_tau(np.array([0.25, 0.5], dtype=dtype))
        self.assertTrue(np.array_equal(model.p_lb, [-3.0, -2.0, -1.0]))
        self.assertTrue(np.array_equal(model.p_ub, [1.0, 2.0, 3.0]))
        self.assertTrue(np.array_equal(model.tau_meas, [0.25, 0.5]))
        with self.assertRaises(crocoddyl.Exception):
            model.p_lb = np.zeros(4, dtype=dtype)
        with self.assertRaises(crocoddyl.Exception):
            model.p_ub = np.zeros(4, dtype=dtype)
        with self.assertRaises(crocoddyl.Exception):
            model.update_tau(np.zeros(3, dtype=dtype))

        data = model.createData()
        self.assertEqual(model.create_calls, 1)
        self.assertIsInstance(data, module.DynamicsDataAbstract)
        self.assertTrue(np.array_equal(data.vdot, np.ones(2, dtype=dtype)))
        self.assertIsNone(data.shared)
        with self.assertRaises(AttributeError):
            data.shared = None
        self.assertFalse(hasattr(data, "tmp_ustatic"))
        fallback_data = module.DynamicsModelAbstract.createData(model)
        self.assertIsInstance(fallback_data, module.DynamicsDataAbstract)
        self.assertTrue(np.array_equal(fallback_data.vdot, np.zeros(2, dtype=dtype)))

        x = np.array([0.1, 0.2, 0.3, 0.4], dtype=dtype)
        u = np.array([0.5, 0.6], dtype=dtype)
        model.calc(data, x, u)
        self.assertEqual(model.calc_calls, 1)
        self.assertEqual(model.calc_terminal, [False])
        self.assertTrue(np.allclose(data.vdot, x.sum() + u.sum()))
        module.DynamicsModelAbstract.calc(model, data, x)
        self.assertEqual(model.calc_calls, 2)
        self.assertEqual(model.calc_terminal, [False, True])
        self.assertTrue(np.allclose(data.vdot, x.sum()))

        model.calcDiff(data, x, u)
        self.assertEqual((model.xu_calls, model.p_calls), (1, 1))
        self.assertEqual(model.xu_terminal, [False])
        self.assertTrue(np.array_equal(data.Fx, np.ones((2, 4), dtype=dtype)))
        self.assertTrue(np.array_equal(data.Fu, np.eye(2, dtype=dtype)))
        self.assertTrue(np.array_equal(data.Fp, np.full((2, 3), 7.0, dtype=dtype)))
        self.assertTrue(np.array_equal(data.dP_dv, np.full(2, 2.0)))
        self.assertTrue(np.array_equal(data.dP_dp, np.full(3, 8.0)))
        self.assertTrue(np.array_equal(data.Hp, np.full(3, 9.0)))
        self.assertTrue(np.array_equal(data.Gp, np.full((2, 3), 10.0)))

        model.calcDiff(data, x)
        self.assertEqual((model.xu_calls, model.p_calls), (2, 1))
        self.assertEqual(model.xu_terminal, [False, True])
        with self.assertRaises(crocoddyl.Exception):
            module.DynamicsModelAbstract.calcDiff_p(model, data, x, u)
        with self.assertRaises(crocoddyl.Exception):
            model.calcDiff(data, np.zeros(5, dtype=dtype), u)
        with self.assertRaises(crocoddyl.Exception):
            model.calcDiff(data, x, np.zeros(3, dtype=dtype))

        copied_data = copy.deepcopy(data)
        for name in (
            "vdot",
            "Fx",
            "Fu",
            "Fp",
            "dissipative_P",
            "dP_dv",
            "dP_dp",
            "h",
            "Hx",
            "Hu",
            "Hp",
            "g",
            "Gx",
            "Gu",
            "Gp",
        ):
            self.assertTrue(
                np.array_equal(getattr(copied_data, name), getattr(data, name))
            )

        p = np.arange(model.np, dtype=dtype)
        module.DynamicsModelAbstract.update_p(model, data, p)
        self.assertEqual(model.update_calls, 1)
        self.assertTrue(np.array_equal(model.last_p, p))
        self.assertTrue(hasattr(model, "set_params"))
        self.assertFalse(hasattr(model, "__copy__"))
        self.assertFalse(hasattr(model, "__deepcopy__"))

        bare = module.DynamicsModelAbstract(module.StateVector(4), model.dyn_type, 3, 2)
        bare_data = bare.createData()
        with self.assertRaises(crocoddyl.Exception):
            module.DynamicsModelAbstract.update_p(
                bare, bare_data, np.zeros(bare.np, dtype=dtype)
            )
        with self.assertRaises(RuntimeError):
            bare.calc(bare_data, x, u)
        with self.assertRaises(RuntimeError):
            bare.calcDiff_xu(bare_data, x, u)

    def check_quasi_static(self, module, dtype, dynamics_type):
        class QuasiStaticProbe(module.DynamicsModelAbstract):
            def __init__(self):
                super().__init__(
                    module.StateVector(4), dynamics_type.ContinuousControl, 0, 2
                )
                self.calc_calls = 0
                self.xu_calls = 0

            def calc(self, data, x, u=None):
                self.assert_running(u)
                data.vdot = u**3 + u + x[:2]
                self.calc_calls += 1

            def calcDiff_xu(self, data, x, u=None):
                self.assert_running(u)
                data.Fu = np.diag(1.0 + 3.0 * u**2).astype(dtype)
                self.xu_calls += 1

            @staticmethod
            def assert_running(u):
                if u is None:
                    raise AssertionError("quasi-static must use running dynamics")

        class QuasiStaticOverride(QuasiStaticProbe):
            def __init__(self):
                super().__init__()
                self.quasi_calls = 0
                self.invalid_result = False

            def quasiStatic(self, data, x, maxiter, tol):
                self.quasi_calls += 1
                size = 3 if self.invalid_result else self.nu
                return np.arange(1, size + 1, dtype=dtype)

        x = np.array([0.5, -0.25, 0.0, 0.0], dtype=dtype)
        tol = 1e-6 if dtype == np.float32 else 1e-12
        fallback = QuasiStaticProbe()
        fallback_data = fallback.createData()
        u = module.DynamicsModelAbstract.quasiStatic(
            fallback, fallback_data, x, 100, tol
        )
        self.assertFalse(np.allclose(u, 0.0))
        self.assertGreater(fallback.calc_calls, 1)
        self.assertGreater(fallback.xu_calls, 1)
        fallback.calc(fallback_data, x, u)
        self.assertLess(np.linalg.norm(fallback_data.vdot), 10.0 * tol)

        overridden = QuasiStaticOverride()
        overridden_data = overridden.createData()
        overridden_u = module.DynamicsModelAbstract.quasiStatic(
            overridden, overridden_data, x, 10, tol
        )
        self.assertTrue(np.array_equal(overridden_u, [1.0, 2.0]))
        self.assertEqual(overridden.quasi_calls, 1)
        overridden.invalid_result = True
        with self.assertRaises(crocoddyl.Exception):
            module.DynamicsModelAbstract.quasiStatic(
                overridden, overridden_data, x, 10, tol
            )

    def check_layouts(self, module, dtype, dynamics_type):
        expected = (
            (dynamics_type.ContinuousControl, (2,), (2, 4)),
            (dynamics_type.ContinuousEstimation, (2,), (2, 4)),
            (dynamics_type.DiscreteTime, (4,), (4, 4)),
        )
        for model_type, vdot_shape, fx_shape in expected:
            model = make_probe(module, dtype, model_type)
            data = module.DynamicsModelAbstract.createData(model)
            self.assertEqual(data.vdot.shape, vdot_shape)
            self.assertEqual(data.Fx.shape, fx_shape)
            self.assertEqual(data.Fu.shape, (fx_shape[0], 2))
            self.assertEqual(data.Fp.shape, (fx_shape[0], 3))
            self.assertEqual(data.dissipative_P.shape, (1,))
            self.assertEqual(data.dP_dv.shape, (2,))
            self.assertEqual(data.dP_dp.shape, (3,))
            self.assertEqual(data.h.shape, (1,))
            self.assertEqual(data.Hx.shape, (4,))
            self.assertEqual(data.Hu.shape, (2,))
            self.assertEqual(data.Hp.shape, (3,))
            self.assertEqual(data.g.shape, (2,))
            self.assertEqual(data.Gx.shape, (2, 4))
            self.assertEqual(data.Gu.shape, (2, 2))
            self.assertEqual(data.Gp.shape, (2, 3))
            for name in (
                "vdot",
                "Fx",
                "Fu",
                "Fp",
                "dissipative_P",
                "dP_dv",
                "dP_dp",
                "h",
                "Hx",
                "Hu",
                "Hp",
                "g",
                "Gx",
                "Gu",
                "Gp",
            ):
                self.assertTrue(np.allclose(getattr(data, name), 0.0))

    def test_float64_contracts(self):
        self.check_model_data_and_dispatch(
            crocoddyl, np.float64, crocoddyl.DynamicsType
        )
        self.check_layouts(crocoddyl, np.float64, crocoddyl.DynamicsType)
        self.check_quasi_static(crocoddyl, np.float64, crocoddyl.DynamicsType)

    def test_float32_contracts(self):
        self.check_model_data_and_dispatch(
            crocoddyl_float32, np.float32, crocoddyl.DynamicsType
        )
        self.check_layouts(crocoddyl_float32, np.float32, crocoddyl.DynamicsType)
        self.check_quasi_static(crocoddyl_float32, np.float32, crocoddyl.DynamicsType)


if __name__ == "__main__":
    unittest.main()
