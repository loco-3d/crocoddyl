import sys
import unittest
from random import randint

import example_robot_data
import numpy as np

import crocoddyl
from crocoddyl.utils import StateMultibodyDerived, StateVectorDerived


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
        self.assertEqual(self.STATE.zero().shape, (self.NX, ), "Wrong dimension of zero state.")
        self.assertEqual(self.STATE.rand().shape, (self.NX, ), "Wrong dimension of random state.")
        self.assertEqual(self.STATE.lb.shape, (self.NX, ), "Wrong dimension of lower bound.")
        self.assertEqual(self.STATE.ub.shape, (self.NX, ), "Wrong dimension of lower bound.")

    def test_python_derived_diff(self):
        x0 = self.STATE.rand()
        x1 = self.STATE.rand()

        # Checking that both diff functions agree
        self.assertTrue(np.allclose(self.STATE.diff(x0, x1), self.STATE_DER.diff(x0, x1), atol=1e-9),
                        "state.diff() function doesn't agree with Python bindings.")

    def test_python_derived_integrate(self):
        x = self.STATE.rand()
        dx = self.STATE.rand()[:self.STATE.ndx]

        # Checking that both integrate functions agree
        self.assertTrue(np.allclose(self.STATE.integrate(x, dx), self.STATE_DER.integrate(x, dx), atol=1e-9),
                        "state.integrate() function doesn't agree with Python bindings.")

    def test_python_derived_Jdiff(self):
        x0 = self.STATE.rand()
        x1 = self.STATE.rand()

        # Checking that both Jdiff functions agree
        J1, J2 = self.STATE.Jdiff(x0, x1)
        J1d, J2d = self.STATE_DER.Jdiff(x0, x1)
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
        J1, J2 = self.STATE.Jintegrate(x, dx)
        J1d, J2d = self.STATE_DER.Jintegrate(x, dx)
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
    NX = randint(1, 101)
    NDX = StateAbstractTestCase.NX
    STATE = crocoddyl.StateVector(NX)
    STATE_DER = StateVectorDerived(NX)


class StateMultibodyTalosArmTest(StateAbstractTestCase):
    MODEL = example_robot_data.load('talos_arm').model
    NX = MODEL.nq + MODEL.nv
    NDX = 2 * MODEL.nv
    STATE = crocoddyl.StateMultibody(MODEL)
    STATE_DER = StateMultibodyDerived(MODEL)


class StateMultibodyHyQTest(StateAbstractTestCase):
    MODEL = example_robot_data.load('hyq').model
    NX = MODEL.nq + MODEL.nv
    NDX = 2 * MODEL.nv
    STATE = crocoddyl.StateMultibody(MODEL)
    STATE_DER = StateMultibodyDerived(MODEL)


class StateMultibodyTalosTest(StateAbstractTestCase):
    MODEL = example_robot_data.load('talos').model
    NX = MODEL.nq + MODEL.nv
    NDX = 2 * MODEL.nv
    STATE = crocoddyl.StateMultibody(MODEL)
    STATE_DER = StateMultibodyDerived(MODEL)


if __name__ == '__main__':
    test_classes_to_run = [StateVectorTest, StateMultibodyTalosArmTest, StateMultibodyHyQTest, StateMultibodyTalosTest]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
    sys.exit(not results.wasSuccessful())
