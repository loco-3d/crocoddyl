import collections
import sys
import unittest

import example_robot_data
import numpy as np

import crocoddyl
import pinocchio
from crocoddyl.utils import Impulse3DModelDerived, Impulse6DModelDerived


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
        self.IMPULSE.calc(self.data, self.x)
        self.IMPULSE.calcDiff(self.data, self.x)

        self.IMPULSE_DER.calc(self.data_der, self.x)
        self.IMPULSE_DER.calcDiff(self.data_der, self.x)
        # Checking the Jacobians of the contact constraint
        self.assertTrue(np.allclose(self.data.dv0_dq, self.data_der.dv0_dq, atol=1e-9),
                        "Wrong Jacobian of the acceleration before impulse (dv0_dq).")


class ImpulseModelMultipleAbstractTestCase(unittest.TestCase):
    ROBOT_MODEL = None
    ROBOT_STATE = None
    IMPULSES = None

    def setUp(self):
        self.x = self.ROBOT_STATE.rand()
        self.robot_data = self.ROBOT_MODEL.createData()

        self.impulseSum = crocoddyl.ImpulseModelMultiple(self.ROBOT_STATE)
        self.datas = collections.OrderedDict([[name, impulse.createData(self.robot_data)]
                                              for name, impulse in self.IMPULSES.items()])
        for name, impulse in self.IMPULSES.items():
            self.impulseSum.addImpulse(name, impulse)
        self.dataSum = self.impulseSum.createData(self.robot_data)

        nq, nv = self.ROBOT_MODEL.nq, self.ROBOT_MODEL.nv
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:],
                                    pinocchio.utils.zero(nv))
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data)
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.computeForwardKinematicsDerivatives(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:],
                                                      pinocchio.utils.zero(nv))

    def test_ni_dimension(self):
        ni = sum([impulse.ni for impulse in self.IMPULSES.values()])
        self.assertEqual(self.impulseSum.ni, ni, "Wrong nc.")

    def test_calc(self):
        # Run calc for both action models
        for impulse, data in zip(self.IMPULSES.values(), self.datas.values()):
            impulse.calc(data, self.x)
        self.impulseSum.calc(self.dataSum, self.x)
        # Checking the cost value and its residual
        Jc = np.vstack([data.Jc for data in self.datas.values()])
        self.assertTrue(np.allclose(self.dataSum.Jc, Jc, atol=1e-9), "Wrong contact Jacobian (Jc).")

    def test_calcDiff(self):
        # Run calc for both action models
        for impulse, data in zip(self.IMPULSES.values(), self.datas.values()):
            impulse.calc(data, self.x)
            impulse.calcDiff(data, self.x)
        self.impulseSum.calc(self.dataSum, self.x)
        self.impulseSum.calcDiff(self.dataSum, self.x)
        # Checking the Jacobians of the contact constraint
        dv0_dq = np.vstack([data.dv0_dq for data in self.datas.values()])
        self.assertTrue(np.allclose(self.dataSum.dv0_dq, dv0_dq, atol=1e-9),
                        "Wrong Jacobian of the velocity before impulse (dv0_dq).")


class Impulse3DTest(ImpulseModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.loadHyQ().model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    # gains = pinocchio.utils.rand(2)
    frame = ROBOT_MODEL.getFrameId('lf_foot')
    IMPULSE = crocoddyl.ImpulseModel3D(ROBOT_STATE, frame)
    IMPULSE_DER = Impulse3DModelDerived(ROBOT_STATE, frame)


class Impulse3DMultipleTest(ImpulseModelMultipleAbstractTestCase):
    ROBOT_MODEL = example_robot_data.loadHyQ().model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    IMPULSES = collections.OrderedDict(
        sorted({
            'lf_foot': crocoddyl.ImpulseModel3D(ROBOT_STATE, ROBOT_MODEL.getFrameId('lf_foot')),
            'rh_foot': crocoddyl.ImpulseModel3D(ROBOT_STATE, ROBOT_MODEL.getFrameId('rh_foot'))
        }.items()))


class Impulse6DTest(ImpulseModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.loadICub().model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    frame = ROBOT_MODEL.getFrameId('r_sole')
    IMPULSE = crocoddyl.ImpulseModel6D(ROBOT_STATE, frame)
    IMPULSE_DER = Impulse6DModelDerived(ROBOT_STATE, frame)


class Impulse6DMultipleTest(ImpulseModelMultipleAbstractTestCase):
    ROBOT_MODEL = example_robot_data.loadICub().model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    IMPULSES = collections.OrderedDict(
        sorted({
            'l_sole': crocoddyl.ImpulseModel6D(ROBOT_STATE, ROBOT_MODEL.getFrameId('l_sole')),
            'r_sole': crocoddyl.ImpulseModel6D(ROBOT_STATE, ROBOT_MODEL.getFrameId('r_sole'))
        }.items()))


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
