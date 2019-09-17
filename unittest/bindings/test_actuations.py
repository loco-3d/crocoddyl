import sys
import unittest

import numpy as np

import crocoddyl
import pinocchio
from crocoddyl.utils import FreeFloatingActuationDerived, FullActuationDerived


class ActuationModelAbstractTestCase(unittest.TestCase):
    STATE = None
    ACTUATION = None
    ACTUATION_DER = None

    def setUp(self):
        self.x = self.STATE.rand()
        self.u = pinocchio.utils.rand(self.ACTUATION.nu)
        self.DATA = self.ACTUATION.createData()
        self.DATA_DER = self.ACTUATION_DER.createData()

    def test_calc(self):
        # Run calc for both action models
        self.ACTUATION.calc(self.DATA, self.x, self.u)
        self.ACTUATION_DER.calc(self.DATA_DER, self.x, self.u)
        # Checking the actuation signal
        self.assertTrue(np.allclose(self.DATA.a, self.DATA_DER.a, atol=1e-9), "Wrong actuation signal.")

    def test_calcDiff(self):
        # Run calcDiff for both action models
        self.ACTUATION.calcDiff(self.DATA, self.x, self.u)
        self.ACTUATION_DER.calcDiff(self.DATA_DER, self.x, self.u)
        # Checking the Jacobians of the actuation model
        self.assertTrue(np.allclose(self.DATA.Ax, self.DATA_DER.Ax, atol=1e-9), "Wrong Ax.")
        self.assertTrue(np.allclose(self.DATA.Au, self.DATA_DER.Au, atol=1e-9), "Wrong Au.")


class FloatingBaseActuationTest(ActuationModelAbstractTestCase):
    STATE = crocoddyl.StateMultibody(pinocchio.buildSampleModelHumanoidRandom())

    ACTUATION = crocoddyl.ActuationModelFloatingBase(STATE)
    ACTUATION_DER = FreeFloatingActuationDerived(STATE)


class FullActuationTest(ActuationModelAbstractTestCase):
    STATE = crocoddyl.StateMultibody(pinocchio.buildSampleModelManipulator())

    ACTUATION = crocoddyl.ActuationModelFull(STATE)
    ACTUATION_DER = FullActuationDerived(STATE)


if __name__ == '__main__':
    test_classes_to_run = [FloatingBaseActuationTest, FullActuationTest]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
    sys.exit(not results.wasSuccessful())
