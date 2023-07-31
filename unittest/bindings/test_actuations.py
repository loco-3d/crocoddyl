import sys
import unittest

import example_robot_data
import numpy as np
import pinocchio
from factory import FreeFloatingActuationDerived, FullActuationDerived

import crocoddyl


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
        self.assertTrue(
            np.allclose(self.DATA.tau, self.DATA_DER.tau, atol=1e-9),
            "Wrong actuation signal.",
        )

    def test_calcDiff(self):
        # Run calcDiff for both action models
        self.ACTUATION.calcDiff(self.DATA, self.x, self.u)
        self.ACTUATION_DER.calcDiff(self.DATA_DER, self.x, self.u)
        # Checking the Jacobians of the actuation model
        self.assertTrue(
            np.allclose(self.DATA.dtau_dx, self.DATA_DER.dtau_dx, atol=1e-9),
            "Wrong dtau_dx.",
        )
        self.assertTrue(
            np.allclose(self.DATA.dtau_du, self.DATA_DER.dtau_du, atol=1e-9),
            "Wrong dtau_du.",
        )


class TalosArmFullActuationTest(ActuationModelAbstractTestCase):
    STATE = crocoddyl.StateMultibody(example_robot_data.load("talos_arm").model)

    ACTUATION = crocoddyl.ActuationModelFull(STATE)
    ACTUATION_DER = FullActuationDerived(STATE)


class HyQFloatingBaseActuationTest(ActuationModelAbstractTestCase):
    STATE = crocoddyl.StateMultibody(example_robot_data.load("hyq").model)

    ACTUATION = crocoddyl.ActuationModelFloatingBase(STATE)
    ACTUATION_DER = FreeFloatingActuationDerived(STATE)


class TalosFloatingBaseActuationTest(ActuationModelAbstractTestCase):
    STATE = crocoddyl.StateMultibody(example_robot_data.load("talos").model)

    ACTUATION = crocoddyl.ActuationModelFloatingBase(STATE)
    ACTUATION_DER = FreeFloatingActuationDerived(STATE)


if __name__ == "__main__":
    # test to be run
    test_classes_to_run = [
        TalosArmFullActuationTest,
        HyQFloatingBaseActuationTest,
        TalosFloatingBaseActuationTest,
    ]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
    sys.exit(not results.wasSuccessful())
