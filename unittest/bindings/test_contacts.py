import sys
import unittest

import numpy as np

import crocoddyl
import pinocchio
from crocoddyl.utils import Contact3DDerived, Contact6DDerived


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


class ContactModelMultipleAbstractTestCase(unittest.TestCase):
    ROBOT_MODEL = None
    ROBOT_STATE = None
    CONTACT = None

    def setUp(self):
        self.x = self.ROBOT_STATE.rand()
        self.robot_data = self.ROBOT_MODEL.createData()

        self.contacts = crocoddyl.ContactModelMultiple(self.ROBOT_STATE)
        self.contacts.addContact("myContact", self.CONTACT)

        self.data = self.CONTACT.createData(self.robot_data)
        self.data_multiple = self.contacts.createData(self.robot_data)

        nq, nv = self.ROBOT_MODEL.nq, self.ROBOT_MODEL.nv
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:],
                                    pinocchio.utils.zero(nv))
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data)
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.computeForwardKinematicsDerivatives(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:],
                                                      pinocchio.utils.zero(nv))

    def test_nc_dimension(self):
        self.assertEqual(self.CONTACT.nc, self.contacts.nc, "Wrong nc.")

    def test_calc(self):
        # Run calc for both action models
        self.CONTACT.calc(self.data, self.x)
        self.contacts.calc(self.data_multiple, self.x)
        # Checking the cost value and its residual
        self.assertTrue(np.allclose(self.data.Jc, self.data_multiple.Jc, atol=1e-9), "Wrong contact Jacobian (Jc).")
        self.assertTrue(np.allclose(self.data.a0, self.data_multiple.a0, atol=1e-9), "Wrong drift acceleration (a0).")

    def test_calcDiff(self):
        # Run calc for both action models
        self.CONTACT.calcDiff(self.data, self.x, True)
        self.contacts.calcDiff(self.data_multiple, self.x, True)
        # Checking the Jacobians of the contact constraint
        self.assertTrue(np.allclose(self.data.Ax, self.data_multiple.Ax, atol=1e-9),
                        "Wrong derivatives of the contact constraint (Ax).")


class Contact3DTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    xref = crocoddyl.FrameTranslation(ROBOT_MODEL.getFrameId('rleg5_joint'), pinocchio.SE3.Random().translation)
    CONTACT = crocoddyl.ContactModel3D(ROBOT_STATE, xref, gains)
    CONTACT_DER = Contact3DDerived(ROBOT_STATE, xref, gains)


class Contact3DMultipleTest(ContactModelMultipleAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    xref = crocoddyl.FrameTranslation(ROBOT_MODEL.getFrameId('rleg5_joint'), pinocchio.SE3.Random().translation)
    CONTACT = crocoddyl.ContactModel3D(ROBOT_STATE, xref, gains)


class Contact6DTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    Mref = crocoddyl.FramePlacement(ROBOT_MODEL.getFrameId('rleg5_joint'), pinocchio.SE3.Random())
    CONTACT = crocoddyl.ContactModel6D(ROBOT_STATE, Mref, gains)
    CONTACT_DER = Contact6DDerived(ROBOT_STATE, Mref, gains)


class Contact6DMultipleTest(ContactModelMultipleAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    Mref = crocoddyl.FramePlacement(ROBOT_MODEL.getFrameId('rleg5_joint'), pinocchio.SE3.Random())
    CONTACT = crocoddyl.ContactModel6D(ROBOT_STATE, Mref, gains)


if __name__ == '__main__':
    test_classes_to_run = [Contact3DTest, Contact3DMultipleTest, Contact6DTest, Contact6DMultipleTest]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
    sys.exit(not results.wasSuccessful())
