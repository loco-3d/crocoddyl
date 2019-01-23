import unittest
from crocoddyl import StateNumDiff
from crocoddyl import StateVector
from crocoddyl import StateUnicycle
from crocoddyl import StatePinocchio
from random import randint
import numpy as np



class StateTestCase(unittest.TestCase):
    NX = None
    STATE = None
    STATE_NUMDIFF = None

    def test_state_dimension(self):
        # Checking the dimension of zero and random states
        self.assertEqual(self.STATE.zero().shape, (self.NX,), \
            "Wrong dimension of zero state.")
        self.assertEqual(self.STATE.rand().shape, (self.NX,), \
            "Wrong dimension of random state.")

    def test_integrate_against_difference(self):
        # Generating random states
        x1 = self.STATE.rand()
        x2 = self.STATE.rand()

        # Computing x2 by integrating its difference
        x2i = self.STATE.integrate(x1,self.STATE.diff(x1,x2))

        # Checking that both states agree
        self.assertTrue(np.allclose(x2i, x2, atol=1e-9), \
            "Integrate function doesn't agree with difference rule.")

    def test_difference_against_integrate(self):
        # Generating random states
        x1 = self.STATE.rand()
        dx = np.random.rand(self.STATE.ndx)

        # Computing dx by differentiation its integrate
        dxd = self.STATE.diff(x1,self.STATE.integrate(x1,dx))

        # Checking that both states agree
        self.assertTrue(np.allclose(dxd, dx, atol=1e-9), \
            "Difference function doesn't agree with integrate rule.")

    def test_Jdiff_against_numdiff(self):
        # Generating random values for the initial and terminal states
        x1 = self.STATE.rand()
        x2 = self.STATE.rand()

        # Computing the partial derivatives of the difference function
        J1,J2 = self.STATE.Jdiff(x1,x2)
        Jnum1,Jnum2 = self.STATE_NUMDIFF.Jdiff(x1,x2)

        # Checking the partial derivatives against NumDiff
        tol = 10*self.STATE_NUMDIFF.disturbance
        self.assertTrue(np.allclose(J1,Jnum1, atol=tol), \
            "The partial derivatives of difference function with respect to first argument is wrong.")
        self.assertTrue(np.allclose(J2,Jnum2, atol=tol), \
            "The partial derivatives of difference function with respect to second argument is wrong.")

    def test_Jintegrate_against_numdiff(self):
        # Generating random values for the initial state and its rate of change
        x = self.STATE.rand()
        vx = np.random.rand(self.STATE.ndx)

        # Computing the partial derivatives of the integrate function
        J1,J2 = self.STATE.Jintegrate(x,vx)
        Jnum1,Jnum2 = self.STATE_NUMDIFF.Jintegrate(x,vx)

        # Checking the partial derivatives against NumDiff
        tol = 10*self.STATE_NUMDIFF.disturbance
        self.assertTrue(np.allclose(J1,Jnum1, atol=tol), \
            "The partial derivatives of integrate function with respect to first argument is wrong.")
        self.assertTrue(np.allclose(J2,Jnum2, atol=tol), \
            "The partial derivatives of integrate function with respect to second argument is wrong.")

    def test_Jdiff_and_Jintegrate_are_inverses(self):
        x1 = self.STATE.rand()
        dx = np.random.rand(self.STATE.ndx)
        x2 = self.STATE.integrate(x1,dx)
        

        # Computing the partial derivatives of the integrate and difference function
        Jx,Jdx = self.STATE.Jintegrate(x1,dx)
        J1,J2 = self.STATE.Jdiff(x1,x2)
        
        # Checking that Jdiff and Jintegrate are inverses
        dX_dDX = Jdx
        dDX_dX = J2
        self.assertTrue(np.allclose(dX_dDX, np.linalg.inv(dDX_dX), atol=1e-9), \
            "Jdiff and Jintegrate aren't inverses.")



class StateVectorTest(StateTestCase):
    StateTestCase.NX = randint(1,101)
    StateTestCase.STATE = StateVector(StateTestCase.NX)
    StateTestCase.STATE_NUMDIFF = StateNumDiff(StateTestCase.STATE)

class StateUnicycleTest(StateTestCase):
    StateTestCase.NX = 3
    StateTestCase.STATE = StateUnicycle()
    StateTestCase.STATE_NUMDIFF = StateNumDiff(StateTestCase.STATE)

class StatePinocchioTest(StateTestCase):
    # Loading Talos arm
    from robots import loadTalosArm
    model = loadTalosArm().model

    StateTestCase.NX = model.nq + model.nv
    StateTestCase.STATE = StatePinocchio(model)
    StateTestCase.STATE_NUMDIFF = StateNumDiff(StateTestCase.STATE)


if __name__ == '__main__':
    unittest.main()