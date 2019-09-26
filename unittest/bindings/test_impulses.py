import sys
import unittest

import numpy as np

import crocoddyl
import pinocchio
from crocoddyl.utils import Impulse3DDerived, Impulse6DDerived


class ImpulseModelAbstractTestCase(unittest.TestCase):
    ROBOT_MODEL = None
    ROBOT_STATE = None
    IMPULSE = None
    IMPULSE_DER = None

    def setUp(self):
        self.x = self.ROBOT_STATE.rand()
        self.robot_data = self.ROBOT_MODEL.createData()
        self.data = self.IMPULSE.createData(self.robot_data)
        self.data_der = self.IMPULSE_DER.createData(self.robot_data)

        nq, nv = self.ROBOT_MODEL.nq, self.ROBOT_MODEL.nv
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:],
                                    pinocchio.utils.zero(nv))
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data)
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.computeForwardKinematicsDerivatives(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:],
                                                      pinocchio.utils.zero(nv))

    def test_ni_dimension(self):
        self.assertEqual(self.IMPULSE.ni, self.IMPULSE_DER.ni, "Wrong ni.")

    def test_calc(self):
        # Run calc for both action models
        self.IMPULSE.calc(self.data, self.x)
        self.IMPULSE_DER.calc(self.data_der, self.x)
        # Checking the cost value and its residual
        self.assertTrue(np.allclose(self.data.Jc, self.data_der.Jc, atol=1e-9), "Wrong contact Jacobian (Jc).")

    def test_calcDiff(self):
        # Run calc for both action models
        self.IMPULSE.calcDiff(self.data, self.x, True)
        self.IMPULSE_DER.calcDiff(self.data_der, self.x, True)
        # Checking the Jacobians of the contact constraint
        self.assertTrue(np.allclose(self.data.dv0_dq, self.data_der.dv0_dq, atol=1e-9),
                        "Wrong Jacobian of the acceleration before impulse (dv0_dq).")


class ImpulseModelMultipleAbstractTestCase(unittest.TestCase):
    ROBOT_MODEL = None
    ROBOT_STATE = None
    IMPULSE = None

    def setUp(self):
        self.x = self.ROBOT_STATE.rand()
        self.robot_data = self.ROBOT_MODEL.createData()

        self.impulses = crocoddyl.ImpulseModelMultiple(self.ROBOT_STATE)
        self.impulses.addImpulse("myImpulse", self.IMPULSE)

        self.data = self.IMPULSE.createData(self.robot_data)
        self.data_multiple = self.impulses.createData(self.robot_data)

        nq, nv = self.ROBOT_MODEL.nq, self.ROBOT_MODEL.nv
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:],
                                    pinocchio.utils.zero(nv))
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data)
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.computeForwardKinematicsDerivatives(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:],
                                                      pinocchio.utils.zero(nv))

    def test_ni_dimension(self):
        self.assertEqual(self.IMPULSE.ni, self.impulses.ni, "Wrong ni.")

    def test_calc(self):
        # Run calc for both action models
        self.IMPULSE.calc(self.data, self.x)
        self.impulses.calc(self.data_multiple, self.x)
        # Checking the cost value and its residual
        self.assertTrue(np.allclose(self.data.Jc, self.data_multiple.Jc, atol=1e-9), "Wrong contact Jacobian (Jc).")

    def test_calcDiff(self):
        # Run calc for both action models
        self.IMPULSE.calcDiff(self.data, self.x, True)
        self.impulses.calcDiff(self.data_multiple, self.x, True)
        # Checking the Jacobians of the contact constraint
        self.assertTrue(np.allclose(self.data.dv0_dq, self.data_multiple.dv0_dq, atol=1e-9),
                        "Wrong Jacobian of the acceleration before impulse (dv0_dq).")


class Impulse3DTest(ImpulseModelAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    # gains = pinocchio.utils.rand(2)
    frame = ROBOT_MODEL.getFrameId('rleg5_joint')
    IMPULSE = crocoddyl.ImpulseModel3D(ROBOT_STATE, frame)
    IMPULSE_DER = Impulse3DDerived(ROBOT_STATE, frame)


class Impulse3DMultipleTest(ImpulseModelMultipleAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    IMPULSE = crocoddyl.ImpulseModel3D(ROBOT_STATE, ROBOT_MODEL.getFrameId('rleg5_joint'))


class Impulse6DTest(ImpulseModelAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    frame = ROBOT_MODEL.getFrameId('rleg5_joint')
    IMPULSE = crocoddyl.ImpulseModel6D(ROBOT_STATE, frame)
    IMPULSE_DER = Impulse6DDerived(ROBOT_STATE, frame)


class Impulse6DMultipleTest(ImpulseModelMultipleAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    IMPULSE = crocoddyl.ImpulseModel6D(ROBOT_STATE, ROBOT_MODEL.getFrameId('rleg5_joint'))


if __name__ == '__main__':
    test_classes_to_run = [Impulse3DTest, Impulse3DMultipleTest, Impulse6DTest, Impulse6DMultipleTest]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
    sys.exit(not results.wasSuccessful())
