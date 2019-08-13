import crocoddyl
import utils
import pinocchio
import numpy as np
import random
import unittest
import sys


class ContactModelAbstractTestCase(unittest.TestCase):
    CONTACT = None
    CONTACT_DER = None

    def setUp(self):
        self.x = self.STATE.rand()
        self.DATA = self.CONTACT.createData(self.ROBOT_DATA)
        self.DATA_DER = self.CONTACT_DER.createData(self.ROBOT_DATA)

        nq = self.ROBOT_MODEL.nq
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.ROBOT_DATA, self.x[:nq], self.x[nq:])
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.ROBOT_DATA, self.x[:nq])
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.ROBOT_DATA)

    def test_nc_dimension(self):
        self.assertEqual(self.CONTACT.nc, self.CONTACT_DER.nc, "Wrong nc.")

    def test_calc(self):
        # Run calc for both action models
        self.CONTACT.calc(self.DATA, self.x)
        self.CONTACT_DER.calc(self.DATA_DER, self.x)
        # Checking the cost value and its residual
        self.assertTrue(np.allclose(self.DATA.Jc, self.DATA_DER.Jc, atol=1e-9), "Wrong contact Jacobian (Jc).")
        self.assertTrue(np.allclose(self.DATA.a0, self.DATA_DER.a0, atol=1e-9), "Wrong drift acceleration (a0).")

    def test_calcDiff(self):
        # Run calc for both action models
        self.CONTACT.calcDiff(self.DATA, self.x, True)
        self.CONTACT_DER.calcDiff(self.DATA_DER, self.x, True)
        # Checking the Jacobians of the contact constraint
        self.assertTrue(np.allclose(self.DATA.Ax, self.DATA_DER.Ax, atol=1e-9),
                        "Wrong derivatives of the contact constraint (Ax).")


class Contact3DTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_DATA = ROBOT_MODEL.createData()
    STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = np.matrix(np.random.rand(2)).T
    xref = crocoddyl.FrameTranslation(ROBOT_MODEL.getFrameId('rleg5_joint'), pinocchio.SE3.Random().translation)
    CONTACT = crocoddyl.ContactModel3D(STATE, xref, gains)
    CONTACT_DER = utils.Contact3DDerived(STATE, xref, gains)


if __name__ == '__main__':
    test_classes_to_run = [Contact3DTest]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
    sys.exit(not results.wasSuccessful())
