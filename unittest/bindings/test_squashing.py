import sys
import unittest

import example_robot_data
import numpy as np

import crocoddyl
import pinocchio
from crocoddyl.utils import (SquashingSmoothSatDerived)

crocoddyl.switchToNumpyMatrix()


class SquashingModelAbstractTestCase(unittest.TestCase):
    SQUASHING = None
    SQUASHING_DER = None

    def setUp(self):
        self.s = pinocchio.utils.rand(self.SQUASHING.ns)
        self.DATA = self.SQUASHING.createData()
        self.DATA_DER = self.SQUASHING_DER.createData()

    def test_calc(self):
        # Run calc for both squashing functions
        self.SQUASHING.calc(self.DATA, self.s)
        self.SQUASHING_DER.calc(self.DATA_DER, self.s)
        # Checking the squashing signal
        self.assertTrue(np.allclose(self.DATA.u, self.DATA_DER.u, atol=1e-9), "Wrong squashing signal.")

    def test_calcDiff(self):
        # Run calcDiff for both squashing functions
        self.SQUASHING.calcDiff(self.DATA, self.s)
        self.SQUASHING_DER.calcDiff(self.DATA_DER, self.s)
        # Checking teh Jacobians of the squashing function
        self.assertTrue(np.allclose(self.DATA.du_ds, self.DATA_DER.du_ds, atol=1e-9), "Wrong ds_du.")


class SmoothSatSquashingTest(SquashingModelAbstractTestCase):
    NS = 4
    U_UB = pinocchio.utils.zero(NS)
    U_LB = pinocchio.utils.zero(NS)
    U_UB.fill(10)
    U_LB.fill(0)
    SQUASHING = crocoddyl.SquashingModelSmoothSat(U_LB, U_UB, NS)
    SQUASHING_DER = SquashingSmoothSatDerived(U_LB, U_UB, NS)


if __name__ == '__main__':
    test_classes_to_run = [SmoothSatSquashingTest]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
    sys.exit(not results.wasSuccessful())
