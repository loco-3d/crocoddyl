import crocoddyl
import numpy as np
import unittest


def a2m(a):
    return np.matrix(a).T

def m2a(m):
    return np.array(m).squeeze()


class UnicyclePyDerived(crocoddyl.ActionModelAbstract):
    def __init__(self):
        crocoddyl.ActionModelAbstract.__init__(self, crocoddyl.StateVector(3), 2, 5)
        self.dt = .1
        self.costWeights = [10., 1.]

    def calc(model, data, x, u=None):
        if u is None:
            u = model.unone
        v, w = m2a(u)
        px, py, theta = m2a(x)
        c, s = np.cos(theta), np.sin(theta)
        # Rollout the dynamics
        data.xnext = a2m([px + c * v * model.dt, py + s * v * model.dt, theta + w * model.dt])
        # Compute the cost value
        data.costResiduals = np.vstack([model.costWeights[0] * x, model.costWeights[1] * u])
        data.cost = .5 * sum(m2a(data.costResiduals)**2)
        return data.xnext, data.cost

    def calcDiff(model, data, x, u=None, recalc=True):
        if u is None:
            u = model.unone
        xnext, cost = model.calc(data, x, u)
        v, w = m2a(u)
        px, py, theta = m2a(x)
        # Cost derivatives
        data.Lx = a2m(m2a(x) * ([model.costWeights[0]**2] * model.nx))
        data.Lu = a2m(m2a(u) * ([model.costWeights[1]**2] * model.nu))
        data.Lxx = np.diag([model.costWeights[0]**2] * model.nx)
        data.Luu = np.diag([model.costWeights[1]**2] * model.nu)
        # Dynamic derivatives
        c, s, dt = np.cos(theta), np.sin(theta), model.dt
        v, w = m2a(u)
        data.Fx = np.matrix([[1, 0, -s * v * dt], [0, 1, c * v * dt], [0, 0, 1]])
        data.Fu = np.matrix([[c * model.dt, 0], [s * model.dt, 0], [0, model.dt]])
        return xnext, cost


class ActionModelAbstractTestCase(unittest.TestCase):
    MODEL = None
    MODEL_DER = None

    def setUp(self):
        self.x = crocoddyl.StateVector(3).rand()
        self.u = np.matrix(np.random.rand(self.MODEL.nu)).T
        self.DATA = self.MODEL.createData()
        self.DATA_DER = self.MODEL_DER.createData()

    def test_calc(self):
        # Run calc for both action models
        self.MODEL.calc(self.DATA, self.x, self.u)
        self.MODEL_DER.calc(self.DATA_DER, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.DATA.cost, self.DATA_DER.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.DATA.costResiduals, self.DATA_DER.costResiduals, atol=1e-9), "Wrong cost residuals.")
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
    ActionModelAbstractTestCase.MODEL_DER = UnicyclePyDerived()


if __name__ == '__main__':
    unittest.main()