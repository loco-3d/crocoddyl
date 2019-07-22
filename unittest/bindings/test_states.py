import crocoddyl
from utils import StateVectorDerived, StateMultibodyDerived
import pinocchio
from random import randint
import numpy as np
import unittest


class StateAbstractTestCase(unittest.TestCase):
    NX = None
    NDX = None
    STATE = None
    STATE_DER = None

    def test_state_dimension(self):
        # Checking the state dimensions
        self.assertEqual(self.STATE.nx, self.STATE_DER.nx, "Wrong nx value.")
        self.assertEqual(self.STATE.ndx, self.STATE_DER.ndx, "Wrong ndx value.")
        self.assertEqual(self.STATE.nq, self.STATE_DER.nq, "Wrong nq value.")
        self.assertEqual(self.STATE.nv, self.STATE_DER.nv, "Wrong nv value.")

        # Checking the dimension of zero and random states
        self.assertEqual(self.STATE.zero().shape, (self.NX, 1), "Wrong dimension of zero state.")
        self.assertEqual(self.STATE.rand().shape, (self.NX, 1), "Wrong dimension of random state.")

    def test_python_derived_diff(self):
        x0 = self.STATE.rand()
        x1 = self.STATE.rand()

        # Checking that both diff functions agree
        if self.STATE.__class__ == crocoddyl.libcrocoddyl_pywrap.StateMultibody:
            self.assertTrue(np.allclose(self.STATE.diff(x0, x1)[3:], self.STATE_DER.diff(x0, x1)[3:], atol=1e-9),
                            "state.diff() function doesn't agree with Python bindings.")
        else:
            self.assertTrue(np.allclose(self.STATE.diff(x0, x1), self.STATE_DER.diff(x0, x1), atol=1e-9),
                            "state.diff() function doesn't agree with Python bindings.")

    def test_python_derived_integrate(self):
        x = self.STATE.rand()
        dx = self.STATE.rand()[:self.STATE.ndx]

        # Checking that both integrate functions agree
        if self.STATE.__class__ == crocoddyl.libcrocoddyl_pywrap.StateMultibody:
            self.assertTrue(np.allclose(self.STATE.integrate(x, dx)[3:], self.STATE_DER.integrate(x, dx)[3:], atol=1e-9),
                            "state.integrate() function doesn't agree with Python bindings.")
        else:
            self.assertTrue(np.allclose(self.STATE.integrate(x, dx), self.STATE_DER.integrate(x, dx), atol=1e-9),
                            "state.integrate() function doesn't agree with Python bindings.")

    def test_python_derived_Jdiff(self):
        x0 = self.STATE.rand()
        x1 = self.STATE.rand()

        # Checking that both Jdiff functions agree
        J1, J2 = self.STATE.Jdiff(x0, x1, "both")
        J1d, J2d = self.STATE_DER.Jdiff(x0, x1, "both")
        if self.STATE.__class__ == crocoddyl.libcrocoddyl_pywrap.StateMultibody:
            nv = self.STATE.nv
            self.assertTrue(np.allclose(J1[nv:, nv:], J1d[nv:, nv:], atol=1e-9),
                            "state.Jdiff()[0] function doesn't agree with Python bindings.")
            self.assertTrue(np.allclose(J2[nv:, nv:], J2d[nv:, nv:], atol=1e-9),
                            "state.Jdiff()[1] function doesn't agree with Python bindings.")
        else:
            self.assertTrue(np.allclose(J1, J1d, atol=1e-9),
                            "state.Jdiff()[0] function doesn't agree with Python bindings.")
            self.assertTrue(np.allclose(J2, J2d, atol=1e-9),
                            "state.Jdiff()[1] function doesn't agree with Python bindings.")

    def test_python_derived_Jintegrate(self):
        x = self.STATE.rand()
        dx = self.STATE.rand()[:self.STATE.ndx]

        # Checking that both Jintegrate functions agree
        J1, J2 = self.STATE.Jintegrate(x, dx, "both")
        J1d, J2d = self.STATE_DER.Jintegrate(x, dx, "both")
        if self.STATE.__class__ == crocoddyl.libcrocoddyl_pywrap.StateMultibody:
            nv = self.STATE.nv
            self.assertTrue(np.allclose(J1[nv:, nv:], J1d[nv:, nv:], atol=1e-9),
                            "state.Jintegrate()[0] function doesn't agree with Python bindings.")
            self.assertTrue(np.allclose(J2[nv:, nv:], J2d[nv:, nv:], atol=1e-9),
                            "state.Jintegrate()[1] function doesn't agree with Python bindings.")
        else:
            self.assertTrue(np.allclose(J1, J1d, atol=1e-9),
                            "state.Jintegrate()[0] function doesn't agree with Python bindings.")
            self.assertTrue(np.allclose(J2, J2d, atol=1e-9),
                            "state.Jintegrate()[1] function doesn't agree with Python bindings.")


class StateVectorTest(StateAbstractTestCase):
    StateAbstractTestCase.NX = randint(1, 101)
    StateAbstractTestCase.NDX = StateAbstractTestCase.NX
    StateAbstractTestCase.STATE = crocoddyl.StateVector(StateAbstractTestCase.NX)
    StateAbstractTestCase.STATE_DER = StateVectorDerived(StateAbstractTestCase.NX)

class StateMultibodyTest(StateAbstractTestCase):
    MODEL = pinocchio.buildSampleModelHumanoid()
    StateAbstractTestCase.NX = MODEL.nq + MODEL.nv
    StateAbstractTestCase.NDX = 2 * MODEL.nv
    StateAbstractTestCase.STATE = crocoddyl.StateMultibody(MODEL)
    StateAbstractTestCase.STATE_DER = StateMultibodyDerived(MODEL)


if __name__ == '__main__':
    unittest.main()
