import crocoddyl
from utils import UnicycleDerived
from random import randint
import numpy as np
import unittest


class ShootingProblemTestCase(unittest.TestCase):
    MODEL = None
    MODEL_DER = None

    def setUp(self):
        self.T = randint(1, 101)
        state = self.MODEL.State
        self.xs = []
        self.us = []
        self.xs.append(state.rand())
        for i in range(self.T):
            self.xs.append(state.rand())
            self.us.append(np.matrix(np.random.rand(self.MODEL.nu)).T)
        self.PROBLEM = crocoddyl.ShootingProblem(self.xs[0], [self.MODEL]*self.T, self.MODEL)
        self.PROBLEM_DER = crocoddyl.ShootingProblem(self.xs[0], [self.MODEL_DER]*self.T, self.MODEL_DER)

    def test_number_of_nodes(self):
        self.assertEqual(self.T, self.PROBLEM.T, "Wrong number of nodes")
    
    def test_calc(self):
        # Running calc functions
        cost = self.PROBLEM.calc(self.xs, self.us)
        costDer = self.PROBLEM_DER.calc(self.xs, self.us)
        self.assertAlmostEqual(cost, costDer, 10, "Wrong cost value")
        for d1, d2 in zip(self.PROBLEM.runningDatas, self.PROBLEM_DER.runningDatas):
            self.assertTrue(np.allclose(d1.xnext, d2.xnext, atol=1e-9), "Next state doesn't match.")

    def test_calcDiff(self):
        # Running calc functions
        cost = self.PROBLEM.calcDiff(self.xs, self.us)
        costDer = self.PROBLEM_DER.calcDiff(self.xs, self.us)
        self.assertAlmostEqual(cost, costDer, 10, "Wrong cost value")
        for d1, d2 in zip(self.PROBLEM.runningDatas, self.PROBLEM_DER.runningDatas):
            self.assertTrue(np.allclose(d1.xnext, d2.xnext, atol=1e-9), "Next state doesn't match.")
            self.assertTrue(np.allclose(d1.Lx, d2.Lx, atol=1e-9), "Lx doesn't match.")
            self.assertTrue(np.allclose(d1.Lu, d2.Lu, atol=1e-9), "Lu doesn't match.")
            self.assertTrue(np.allclose(d1.Lxx, d2.Lxx, atol=1e-9), "Lxx doesn't match.")
            self.assertTrue(np.allclose(d1.Lxu, d2.Lxu, atol=1e-9), "Lxu doesn't match.")
            self.assertTrue(np.allclose(d1.Luu, d2.Luu, atol=1e-9), "Luu doesn't match.")
            self.assertTrue(np.allclose(d1.Fx, d2.Fx, atol=1e-9), "Fx doesn't match.")
            self.assertTrue(np.allclose(d1.Fu, d2.Fu, atol=1e-9), "Fu doesn't match.")

    def test_rollout(self):
        xs = self.PROBLEM.rollout(self.us)
        xsDer = self.PROBLEM_DER.rollout(self.us)
        for x1, x2 in zip(xs, xsDer):
            self.assertTrue(np.allclose(x1, x2, atol=1e-9), "The rollout state doesn't match.")


class UnicycleShootingTest(ShootingProblemTestCase):
    ShootingProblemTestCase.MODEL = crocoddyl.ActionModelUnicycle()
    ShootingProblemTestCase.MODEL_DER = UnicycleDerived()

if __name__ == '__main__':
    unittest.main()
