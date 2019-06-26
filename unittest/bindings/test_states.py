import crocoddyl
from random import randint
import numpy as np
import unittest


class StateVectorPyDerived(crocoddyl.StateAbstract):
    def __init__(self, nx):
        crocoddyl.StateAbstract.__init__(self, nx, nx)

    def zero(self):
        return np.matrix(np.zeros(self.nx)).T

    def rand(self):
        return np.matrix(np.random.rand(self.nx)).T

    def diff(self, x0, x1):
        dx = x1 - x0
        return x1 - x0

    def integrate(self, x, dx):
        return x + dx

    def Jdiff(self, x1, x2, firstsecond='both'):
        assert (firstsecond in ['first', 'second', 'both'])
        if firstsecond == 'both':
            return [self.Jdiff(x1, x2, 'first'), self.Jdiff(x1, x2, 'second')]

        J = np.zeros([self.ndx, self.ndx])
        if firstsecond == 'first':
            J[:, :] = -np.eye(self.ndx)
        elif firstsecond == 'second':
            J[:, :] = np.eye(self.ndx)
        return J

    def Jintegrate(self, x, dx, firstsecond='both'):
        assert (firstsecond in ['first', 'second', 'both'])
        if firstsecond == 'both':
            return [self.Jintegrate(x, dx, 'first'), self.Jintegrate(x, dx, 'second')]
        return np.eye(self.ndx)


class StateAbstractTestCase(unittest.TestCase):
    NX = None
    NDX = None
    STATE = None
    STATE_DER = None

    def test_state_dimension(self):
        # Checking the state dimensions
        self.assertEqual(self.STATE.nx, self.STATE_DER.nx, "Wrong nx value.")
        self.assertEqual(self.STATE.ndx, self.STATE_DER.ndx, "Wrong ndx value.")

        # Checking the dimension of zero and random states
        self.assertEqual(self.STATE.zero().shape, (self.NX, 1), "Wrong dimension of zero state.")
        self.assertEqual(self.STATE.rand().shape, (self.NX, 1), "Wrong dimension of random state.")

    def test_python_derived_diff(self):
        x0 = self.STATE.rand()
        x1 = self.STATE.rand()

        # Checking that both diff functions agree
        self.assertTrue(
            np.allclose(self.STATE.diff(x0, x1), self.STATE_DER.diff(x0, x1), atol=1e-9),
            "state.diff() function doesn't agree with Python bindings.")

    def test_python_derived_integrate(self):
        x = self.STATE.rand()
        dx = self.STATE.rand()[:self.STATE.ndx]

        # Checking that both integrate functions agree
        self.assertTrue(
            np.allclose(self.STATE.integrate(x, dx), self.STATE_DER.integrate(x, dx), atol=1e-9),
            "state.integrate() function doesn't agree with Python bindings.")

    def test_python_derived_Jdiff(self):
        x0 = self.STATE.rand()
        x1 = self.STATE.rand()

        # Checking that both Jdiff functions agree
        J1, J2 = self.STATE.Jdiff(x0, x1, "both")
        J1d, J2d = self.STATE_DER.Jdiff(x0, x1, "both")
        self.assertTrue(
            np.allclose(J1, J1d, atol=1e-9), "state.Jdiff()[0] function doesn't agree with Python bindings.")
        self.assertTrue(
            np.allclose(J2, J2d, atol=1e-9), "state.Jdiff()[1] function doesn't agree with Python bindings.")

    def test_python_derived_Jintegrate(self):
        x = self.STATE.rand()
        dx = self.STATE.rand()[:self.STATE.ndx]

        # Checking that both Jintegrate functions agree
        J1, J2 = self.STATE.Jintegrate(x, dx, "both")
        J1d, J2d = self.STATE_DER.Jintegrate(x, dx, "both")
        self.assertTrue(
            np.allclose(J1, J1d, atol=1e-9), "state.Jintegrate()[0] function doesn't agree with Python bindings.")
        self.assertTrue(
            np.allclose(J2, J2d, atol=1e-9), "state.Jintegrate()[1] function doesn't agree with Python bindings.")


class StateVectorTest(StateAbstractTestCase):
    StateAbstractTestCase.NX = randint(1, 101)
    StateAbstractTestCase.NDX = StateAbstractTestCase.NX
    StateAbstractTestCase.STATE = crocoddyl.StateVector(StateAbstractTestCase.NX)
    StateAbstractTestCase.STATE_DER = StateVectorPyDerived(StateAbstractTestCase.NX)


if __name__ == '__main__':
    unittest.main()
