import unittest
from crocoddyl import StateVector, StatePinocchio
from random import randint
import numpy as np



class StateTestCase(unittest.TestCase):
    STATE = None
    NX = None

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



class StateVectorTest(StateTestCase):
    StateTestCase.NX = randint(1,101)
    StateTestCase.STATE = StateVector(StateTestCase.NX)

class StatePinocchioTest(StateTestCase):
    # Loading Talos arm
    from robots import loadTalosArm
    model = loadTalosArm().model

    StateTestCase.NX = model.nq + model.nv
    StateTestCase.STATE = StatePinocchio(model)



if __name__ == '__main__':
    unittest.main()