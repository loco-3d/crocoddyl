import unittest
from crocoddyl import ActionModelUnicycle, ActionModelUnicycleVar
from crocoddyl import StateNumDiff, ActionModelNumDiff
import numpy as np


class ActionUnicycleTest(unittest.TestCase):
    MODEL = ActionModelUnicycle()

    def setUp(self):
        # Creating Unicycle action model
        self.model = self.MODEL
        self.mnum = ActionModelNumDiff(self.model, withGaussApprox=True)

        # Creating the Unicycle data
        self.data = self.model.createData()
        self.dnum = self.mnum.createData()

    def test_calc_retunrs_state(self):
        # Generating random state and control vectors
        x = self.model.State.rand()
        u = np.random.rand(self.model.nu)

        # Getting the state dimension from calc() call
        nx = self.model.calc(self.data, x, u)[0].shape

        # Checking the dimension for the state and its tangent
        self.assertEqual(nx, (self.model.nx,), \
            "Dimension of state vector is wrong.")

    def test_calc_returns_a_cost(self):
        # Getting the cost value computed by calc()
        x = self.model.State.rand()
        u = np.random.rand(self.model.nu)
        cost = self.model.calc(self.data, x, u)[1]

        # Checking that calc returns a cost value
        self.assertTrue(isinstance(cost,float), \
            "calc() doesn't return a cost value.")

    def test_partial_derivatives_against_numdiff(self):
        # Generating random values for the state and control
        x = self.model.State.rand()
        u = np.random.rand(self.model.nu)

        # Computing the action derivatives
        self.model.calcDiff(self.data,x,u)
        self.mnum.calcDiff(self.dnum,x,u)

        # Checking the partial derivatives against NumDiff
        tol = 10*self.mnum.disturbance
        self.assertTrue(np.allclose(self.data.Fx,self.dnum.Fx, atol=tol), \
            "Fx is wrong.")
        self.assertTrue(np.allclose(self.data.Fu,self.dnum.Fu, atol=tol), \
            "Fu is wrong.")
        self.assertTrue(np.allclose(self.data.Lx,self.dnum.Lx, atol=tol), \
            "Fx is wrong.")
        self.assertTrue(np.allclose(self.data.Lu,self.dnum.Lu, atol=tol), \
            "Fx is wrong.")
        self.assertTrue(np.allclose(self.data.Lxx,self.dnum.Lxx, atol=tol), \
            "Fx is wrong.")
        self.assertTrue(np.allclose(self.data.Lxu,self.dnum.Lxu, atol=tol), \
            "Fx is wrong.")
        self.assertTrue(np.allclose(self.data.Luu,self.dnum.Luu, atol=tol), \
            "Fx is wrong.")



class ActionUnicycleVarTest(ActionUnicycleTest):
    MODEL = ActionModelUnicycleVar()

    def test_rollout_against_unicycle(self):
        # Creating the Unycicle action model
        X = self.model.State
        model0 = ActionModelUnicycle()
        data0 = model0.createData()

        # Generating random values for the state and control vectors
        x = X.rand()
        x0 = X.diff(X.zero(),x)
        u = np.random.rand(self.model.nu)

        # Making the rollout
        xnext,cost = self.model.calc(self.data,x,u)
        xnext0,cost0 = model0.calc(data0,x0,u)

        # Checking the rollout (next state) and cost values
        self.assertTrue(
            np.allclose(X.integrate(X.zero(),xnext0), xnext, atol=1e-9), \
            "Dynamics simulation is wrong.")
        self.assertAlmostEqual(cost0, cost, "Cost computation is wrong.")



if __name__ == '__main__':
    unittest.main()