import sys
import unittest
from random import randint

import numpy as np
from factory import LQRModelDerived, UnicycleModelDerived

import crocoddyl


class ActionModelParameterDerived(crocoddyl.ActionModelAbstract):
    def __init__(self, np_=3):
        super().__init__(crocoddyl.StateVector(4), 2, 3, 2, 2, 5, 4, np_)
        self.last_p = None

    def calc(self, data, x, u=None):
        data.xnext[:] = x
        data.cost = 0.0

    def calcDiff(self, data, x, u=None):
        pass

    def update_p(self, data, p):
        self.last_p = np.array(p, copy=True)


class ActionModelParameterDefault(crocoddyl.ActionModelAbstract):
    def __init__(self, np_=3):
        super().__init__(crocoddyl.StateVector(4), 2, 3, 2, 1, 0, 0, np_)

    def calc(self, data, x, u=None):
        data.xnext[:] = x
        data.cost = 0.0

    def calcDiff(self, data, x, u=None):
        pass


class ActionBaseParametersTest(unittest.TestCase):
    def test_constructor_and_data_fields(self):
        model = ActionModelParameterDerived()
        data = model.createData()

        self.assertEqual(model.nr, 3)
        self.assertEqual(model.np, 3)
        self.assertEqual(data.Fp.shape, (model.state.ndx, model.np))
        self.assertEqual(data.Lp.shape, (model.np,))
        self.assertEqual(data.Lpp.shape, (model.np, model.np))
        self.assertEqual(data.Lpx.shape, (model.np, model.state.ndx))
        self.assertEqual(data.Lpu.shape, (model.np, model.nu))
        self.assertEqual(data.Gp.shape, (model.ng, model.np))
        self.assertEqual(data.Hp.shape, (model.nh, model.np))
        self.assertFalse(hasattr(data, "dissipative_E"))
        self.assertFalse(hasattr(data, "dE_dv"))
        self.assertFalse(hasattr(data, "dE_dp"))
        self.assertFalse(hasattr(data, "resize"))
        self.assertFalse(hasattr(data, "setZero"))

    def test_update_p_hook_dispatch(self):
        model = ActionModelParameterDerived()
        data = model.createData()
        p = np.arange(model.np, dtype=float)

        crocoddyl.ActionModelAbstract.update_p(model, data, p)
        self.assertTrue(np.array_equal(model.last_p, p))

    def test_default_update_p(self):
        model = ActionModelParameterDefault()
        data = model.createData()

        with self.assertRaises(crocoddyl.Exception):
            crocoddyl.ActionModelAbstract.update_p(model, data, np.zeros(model.np))

    def test_parameter_field_assignment(self):
        model = ActionModelParameterDerived()
        data = model.createData()
        data.Fp = np.full((model.state.ndx, model.np), 1.0)
        data.Lp = np.full(model.np, 2.0)
        data.Lpp = np.full((model.np, model.np), 3.0)
        data.Lpx = np.full((model.np, model.state.ndx), 4.0)
        data.Lpu = np.full((model.np, model.nu), 5.0)
        data.Gp = np.full((model.ng, model.np), 6.0)
        data.Hp = np.full((model.nh, model.np), 7.0)

        self.assertTrue(np.all(data.Fp == 1.0))
        self.assertTrue(np.all(data.Lp == 2.0))
        self.assertTrue(np.all(data.Lpp == 3.0))
        self.assertTrue(np.all(data.Lpx == 4.0))
        self.assertTrue(np.all(data.Lpu == 5.0))
        self.assertTrue(np.all(data.Gp == 6.0))
        self.assertTrue(np.all(data.Hp == 7.0))

        model.np = 4
        resized_data = model.createData()
        self.assertEqual(resized_data.Fp.shape, (model.state.ndx, model.np))
        self.assertEqual(resized_data.Lp.shape, (model.np,))
        self.assertEqual(resized_data.Lpp.shape, (model.np, model.np))
        self.assertEqual(resized_data.Lpx.shape, (model.np, model.state.ndx))
        self.assertEqual(resized_data.Lpu.shape, (model.np, model.nu))
        self.assertEqual(resized_data.Gp.shape, (model.ng, model.np))
        self.assertEqual(resized_data.Hp.shape, (model.nh, model.np))


class ActionModelAbstractTestCase(unittest.TestCase):
    MODEL = None
    MODEL_DER = None

    def setUp(self):
        rng = np.random.default_rng()
        state = self.MODEL.state
        self.x = state.rand()
        self.u = rng.random(self.MODEL.nu)
        self.DATA = self.MODEL.createData()
        self.DATA_DER = self.MODEL_DER.createData()

    def test_calc(self):
        # Run calc for both action models
        self.MODEL.calc(self.DATA, self.x, self.u)
        self.MODEL_DER.calc(self.DATA_DER, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(
            self.DATA.cost, self.DATA_DER.cost, 10, "Wrong cost value."
        )
        self.assertTrue(
            np.allclose(self.DATA.r, self.DATA_DER.r, atol=1e-9),
            "Wrong cost residuals.",
        )

        # Checking the dimension and value of the next state
        self.assertEqual(
            self.DATA.xnext.shape,
            self.DATA_DER.xnext.shape,
            "Wrong next state dimension.",
        )
        self.assertTrue(
            np.allclose(self.DATA.xnext, self.DATA_DER.xnext, atol=1e-9),
            "Wrong next state.",
        )

    def test_calc_x(self):
        # Run calc for both action models
        self.MODEL.calc(self.DATA, self.x)
        self.MODEL_DER.calc(self.DATA_DER, self.x)
        # Checking the cost value and its residual
        self.assertAlmostEqual(
            self.DATA.cost, self.DATA_DER.cost, 10, "Wrong cost value."
        )
        self.assertTrue(
            np.allclose(self.DATA.r, self.DATA_DER.r, atol=1e-9),
            "Wrong cost residuals.",
        )

        # Checking the dimension and value of the next state
        self.assertEqual(
            self.DATA.xnext.shape,
            self.DATA_DER.xnext.shape,
            "Wrong next state dimension.",
        )
        self.assertTrue(
            np.allclose(self.DATA.xnext, self.DATA_DER.xnext, atol=1e-9),
            "Wrong next state.",
        )

    def test_calcDiff(self):
        # Run calcDiff for both action models
        self.MODEL.calc(self.DATA, self.x, self.u)
        self.MODEL.calcDiff(self.DATA, self.x, self.u)

        self.MODEL_DER.calc(self.DATA_DER, self.x, self.u)
        self.MODEL_DER.calcDiff(self.DATA_DER, self.x, self.u)
        # Checking the next state value
        self.assertTrue(
            np.allclose(self.DATA.xnext, self.DATA_DER.xnext, atol=1e-9),
            "Wrong next state.",
        )
        # Checking the Jacobians of the dynamic
        self.assertTrue(
            np.allclose(self.DATA.Fx, self.DATA_DER.Fx, atol=1e-9), "Wrong Fx."
        )
        self.assertTrue(
            np.allclose(self.DATA.Fu, self.DATA_DER.Fu, atol=1e-9), "Wrong Fu."
        )
        # Checking the Jacobians and Hessians of the cost
        self.assertTrue(
            np.allclose(self.DATA.Lx, self.DATA_DER.Lx, atol=1e-9), "Wrong Lx."
        )
        self.assertTrue(
            np.allclose(self.DATA.Lu, self.DATA_DER.Lu, atol=1e-9), "Wrong Lu."
        )
        self.assertTrue(
            np.allclose(self.DATA.Lxx, self.DATA_DER.Lxx, atol=1e-9), "Wrong Lxx."
        )
        self.assertTrue(
            np.allclose(self.DATA.Lxu, self.DATA_DER.Lxu, atol=1e-9), "Wrong Lxu."
        )
        self.assertTrue(
            np.allclose(self.DATA.Luu, self.DATA_DER.Luu, atol=1e-9), "Wrong Luu."
        )

    def test_calcDiff_x(self):
        # Run calcDiff for both action models
        self.MODEL.calc(self.DATA, self.x)
        self.MODEL.calcDiff(self.DATA, self.x)

        self.MODEL_DER.calc(self.DATA_DER, self.x)
        self.MODEL_DER.calcDiff(self.DATA_DER, self.x)
        # Checking the next state value
        self.assertTrue(
            np.allclose(self.DATA.xnext, self.DATA_DER.xnext, atol=1e-9),
            "Wrong next state.",
        )
        # Checking the Jacobians and Hessians of the cost
        self.assertTrue(
            np.allclose(self.DATA.Lx, self.DATA_DER.Lx, atol=1e-9), "Wrong Lx."
        )
        self.assertTrue(
            np.allclose(self.DATA.Lxx, self.DATA_DER.Lxx, atol=1e-9), "Wrong Lxx."
        )

    def test_getters(self):
        # Enforce to run getters
        _, _, _, _ = self.MODEL.ng, self.MODEL.ng_T, self.MODEL.nh, self.MODEL.nh_T


class UnicycleTest(ActionModelAbstractTestCase):
    MODEL = crocoddyl.ActionModelUnicycle()
    MODEL_DER = UnicycleModelDerived()


class LQRTest(ActionModelAbstractTestCase):
    NX = randint(2, 21)
    NU = randint(2, NX)
    MODEL = crocoddyl.ActionModelLQR(NX, NU)
    MODEL_DER = LQRModelDerived(NX, NU)


class RandomLQRTest(ActionModelAbstractTestCase):
    NX = randint(2, 21)
    NU = randint(2, NX)
    MODEL = crocoddyl.ActionModelLQR.Random(NX, NU)
    MODEL_DER = LQRModelDerived.fromLQR(
        MODEL.A, MODEL.B, MODEL.Q, MODEL.R, MODEL.N, MODEL.f, MODEL.q, MODEL.r
    )


if __name__ == "__main__":
    # test to be run
    test_classes_to_run = [
        ActionBaseParametersTest,
        UnicycleTest,
        LQRTest,
        RandomLQRTest,
    ]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
    sys.exit(not results.wasSuccessful())
