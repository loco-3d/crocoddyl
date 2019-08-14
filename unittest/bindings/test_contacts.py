import crocoddyl
import utils
import pinocchio
import numpy as np
import unittest
import sys


class ContactModelAbstractTestCase(unittest.TestCase):
    ROBOT_MODEL = None
    ROBOT_STATE = None
    CONTACT = None
    CONTACT_DER = None

    def setUp(self):
        self.x = self.ROBOT_STATE.rand()
        self.robot_data = self.ROBOT_MODEL.createData()
        self.data = self.CONTACT.createData(self.robot_data)
        self.data_der = self.CONTACT_DER.createData(self.robot_data)

        nq, nv = self.ROBOT_MODEL.nq, self.ROBOT_MODEL.nv
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:],
                                    pinocchio.utils.zero(nv))
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data)
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.computeForwardKinematicsDerivatives(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:],
                                                      pinocchio.utils.zero(nv))

    def test_nc_dimension(self):
        self.assertEqual(self.CONTACT.nc, self.CONTACT_DER.nc, "Wrong nc.")

    def test_calc(self):
        # Run calc for both action models
        self.CONTACT.calc(self.data, self.x)
        self.CONTACT_DER.calc(self.data_der, self.x)
        # Checking the cost value and its residual
        self.assertTrue(np.allclose(self.data.Jc, self.data_der.Jc, atol=1e-9), "Wrong contact Jacobian (Jc).")
        self.assertTrue(np.allclose(self.data.a0, self.data_der.a0, atol=1e-9), "Wrong drift acceleration (a0).")

    def test_calcDiff(self):
        # Run calc for both action models
        self.CONTACT.calcDiff(self.data, self.x, True)
        self.CONTACT_DER.calcDiff(self.data_der, self.x, True)
        # Checking the Jacobians of the contact constraint
        self.assertTrue(np.allclose(self.data.Ax, self.data_der.Ax, atol=1e-9),
                        "Wrong derivatives of the contact constraint (Ax).")


class Contact3DTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = np.matrix(np.random.rand(2)).T
    xref = crocoddyl.FrameTranslation(ROBOT_MODEL.getFrameId('rleg5_joint'), pinocchio.SE3.Random().translation)
    CONTACT = crocoddyl.ContactModel3D(ROBOT_STATE, xref, gains)
    CONTACT_DER = utils.Contact3DDerived(ROBOT_STATE, xref, gains)


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
