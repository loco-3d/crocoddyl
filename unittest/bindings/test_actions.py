import crocoddyl
from utils import UnicycleDerived, LQRDerived
from random import randint
import numpy as np
import unittest


class ActionModelAbstractTestCase(unittest.TestCase):
    MODEL = None
    MODEL_DER = None

    def setUp(self):
        state = self.MODEL.State
        self.x = state.rand()
        self.u = np.matrix(np.random.rand(self.MODEL.nu)).T
        self.DATA = self.MODEL.createData()
        self.DATA_DER = self.MODEL_DER.createData()

    def test_calc(self):
        # Run calc for both action models
        self.MODEL.calc(self.DATA, self.x, self.u)
        self.MODEL_DER.calc(self.DATA_DER, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.DATA.cost, self.DATA_DER.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.DATA.costResiduals, self.DATA_DER.costResiduals, atol=1e-9),
                        "Wrong cost residuals.")
        # Checking the dimension of the next state
        self.assertEqual(self.DATA.xnext.shape, self.DATA_DER.xnext.shape, "Wrong next state dimension.")
        # Checking the next state value
        self.assertTrue(np.allclose(self.DATA.xnext, self.DATA_DER.xnext, atol=1e-9), "Wrong next state.")

    def test_calcDiff(self):
        # Run calcDiff for both action models
        self.MODEL.calcDiff(self.DATA, self.x, self.u)
        self.MODEL_DER.calcDiff(self.DATA_DER, self.x, self.u)
        # Checking the next state value
        self.assertTrue(np.allclose(self.DATA.xnext, self.DATA_DER.xnext, atol=1e-9), "Wrong next state.")
        # Checking the Jacobians of the dynamic
        self.assertTrue(np.allclose(self.DATA.Fx, self.DATA_DER.Fx, atol=1e-9), "Wrong Fx.")
        self.assertTrue(np.allclose(self.DATA.Fu, self.DATA_DER.Fu, atol=1e-9), "Wrong Fu.")
        # Checking the Jacobians and Hessians of the cost
        self.assertTrue(np.allclose(self.DATA.Lx, self.DATA_DER.Lx, atol=1e-9), "Wrong Lx.")
        self.assertTrue(np.allclose(self.DATA.Lu, self.DATA_DER.Lu, atol=1e-9), "Wrong Lu.")
        self.assertTrue(np.allclose(self.DATA.Lxx, self.DATA_DER.Lxx, atol=1e-9), "Wrong Lxx.")
        self.assertTrue(np.allclose(self.DATA.Lxu, self.DATA_DER.Lxu, atol=1e-9), "Wrong Lxu.")
        self.assertTrue(np.allclose(self.DATA.Luu, self.DATA_DER.Luu, atol=1e-9), "Wrong Luu.")


class UnicycleTest(ActionModelAbstractTestCase):
    ActionModelAbstractTestCase.MODEL = crocoddyl.ActionModelUnicycle()
    ActionModelAbstractTestCase.MODEL_DER = UnicycleDerived()


class LQRTest(ActionModelAbstractTestCase):
    NX = randint(1, 21)
    NU = randint(1, NX)
    ActionModelAbstractTestCase.MODEL = crocoddyl.ActionModelLQR(NX, NU)
    ActionModelAbstractTestCase.MODEL_DER = LQRDerived(NX, NU)


if __name__ == '__main__':
    unittest.main()
