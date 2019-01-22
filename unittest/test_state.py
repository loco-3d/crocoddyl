import unittest
from crocoddyl import StateVector
from random import randint
import numpy as np



class StateVectorTest(unittest.TestCase):
    def setUp(self):
        # Creating a random state
        self.NX = randint(1,101)
        self.X = StateVector(self.NX)

    def test_state_dimension(self):
        # Checking the dimension of zero and random states
        self.assertEqual(self.X.zero().shape, (self.NX,), \
            "Wrong dimension of zero state.")
        self.assertEqual(self.X.rand().shape, (self.NX,), \
            "Wrong dimension of random state.")

    def test_integrate_against_difference(self):
        # Generating random states
        x1 = self.X.rand()
        x2 = self.X.rand()

        # Computing the x2 by integrating its difference
        x2i = self.X.integrate(x1,self.X.diff(x1,x2))

        # Checking that both states agree
        self.assertTrue(np.allclose(x2i, x2, atol=1e-9), \
            "Integrate function doesn't agree with difference rule.")


if __name__ == '__main__':
    unittest.main()