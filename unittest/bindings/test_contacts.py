import collections
import sys
import unittest

import example_robot_data
import numpy as np
import pinocchio
from factory import Contact1DModelDerived, Contact3DModelDerived, Contact6DModelDerived

import crocoddyl


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
        pinocchio.forwardKinematics(
            self.ROBOT_MODEL,
            self.robot_data,
            self.x[:nq],
            self.x[nq:],
            pinocchio.utils.zero(nv),
        )
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data)
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.computeForwardKinematicsDerivatives(
            self.ROBOT_MODEL,
            self.robot_data,
            self.x[:nq],
            self.x[nq:],
            pinocchio.utils.zero(nv),
        )

    def test_nc_dimension(self):
        self.assertEqual(self.CONTACT.nc, self.CONTACT_DER.nc, "Wrong nc.")

    def test_calc(self):
        # Run calc for both action models
        self.CONTACT.calc(self.data, self.x)
        self.CONTACT_DER.calc(self.data_der, self.x)
        # Checking the cost value and its residual
        self.assertTrue(
            np.allclose(self.data.Jc, self.data_der.Jc, atol=1e-9),
            "Wrong contact Jacobian (Jc).",
        )
        self.assertTrue(
            np.allclose(self.data.a0, self.data_der.a0, atol=1e-9),
            "Wrong drift acceleration (a0).",
        )

    def test_calcDiff(self):
        # Run calcDiff for both action models
        self.CONTACT.calc(self.data, self.x)
        self.CONTACT.calcDiff(self.data, self.x)
        self.CONTACT_DER.calc(self.data_der, self.x)
        self.CONTACT_DER.calcDiff(self.data_der, self.x)
        # Checking the Jacobians of the contact constraint
        self.assertTrue(
            np.allclose(self.data.da0_dx, self.data_der.da0_dx, atol=1e-9),
            "Wrong derivatives of the desired contact acceleration (da0_dx).",
        )

    def test_updateForce(self):
        # Run updateForce for both action models
        self.CONTACT.calc(self.data, self.x)
        self.CONTACT_DER.calc(self.data_der, self.x)
        rng = np.random.default_rng()
        force = rng.random(self.CONTACT.nc)
        self.CONTACT.updateForce(self.data, force)
        self.CONTACT_DER.updateForce(self.data_der, force)
        self.assertTrue(
            np.allclose(self.data.fext.vector, self.data_der.fext.vector, atol=1e-9),
            "Wrong external spatial force.",
        )


class ContactModelMultipleAbstractTestCase(unittest.TestCase):
    ROBOT_MODEL = None
    ROBOT_STATE = None
    CONTACTS = None

    def setUp(self):
        self.x = self.ROBOT_STATE.rand()
        self.robot_data = self.ROBOT_MODEL.createData()

        self.contactSum = crocoddyl.ContactModelMultiple(self.ROBOT_STATE)
        self.datas = collections.OrderedDict(
            [
                [name, contact.createData(self.robot_data)]
                for name, contact in self.CONTACTS.items()
            ]
        )
        for name, contact in self.CONTACTS.items():
            self.contactSum.addContact(name, contact)
        self.dataSum = self.contactSum.createData(self.robot_data)

        nq, nv = self.ROBOT_MODEL.nq, self.ROBOT_MODEL.nv
        pinocchio.forwardKinematics(
            self.ROBOT_MODEL,
            self.robot_data,
            self.x[:nq],
            self.x[nq:],
            pinocchio.utils.zero(nv),
        )
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data)
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.computeForwardKinematicsDerivatives(
            self.ROBOT_MODEL,
            self.robot_data,
            self.x[:nq],
            self.x[nq:],
            pinocchio.utils.zero(nv),
        )

    def test_nc_dimension(self):
        nc = sum([contact.nc for contact in self.CONTACTS.values()])
        self.assertEqual(self.contactSum.nc, nc, "Wrong nc.")

    def test_calc(self):
        # Run calc for both action models
        for contact, data in zip(self.CONTACTS.values(), self.datas.values()):
            contact.calc(data, self.x)
        self.contactSum.calc(self.dataSum, self.x)
        # Checking the cost value and its residual
        Jc = np.vstack([data.Jc for data in self.datas.values()])
        a0 = np.hstack([data.a0 for data in self.datas.values()])
        self.assertTrue(
            np.allclose(self.dataSum.Jc, Jc, atol=1e-9), "Wrong contact Jacobian (Jc)."
        )
        self.assertTrue(
            np.allclose(self.dataSum.a0, a0, atol=1e-9),
            "Wrong drift acceleration (a0).",
        )

    def test_calcDiff(self):
        # Run calc for both action models
        for contact, data in zip(self.CONTACTS.values(), self.datas.values()):
            contact.calc(data, self.x)
            contact.calcDiff(data, self.x)
        self.contactSum.calc(self.dataSum, self.x)
        self.contactSum.calcDiff(self.dataSum, self.x)
        # Checking the Jacobians of the contact constraint
        da0_dx = np.vstack([data.da0_dx for data in self.datas.values()])
        self.assertTrue(
            np.allclose(self.dataSum.da0_dx, da0_dx, atol=1e-9),
            "Wrong derivatives of the desired contact acceleration (da0_dx).",
        )


class Contact1DLocalTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("hyq").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    xref = pinocchio.SE3.Random().translation[2]
    CONTACT = crocoddyl.ContactModel1D(
        ROBOT_STATE, ROBOT_MODEL.getFrameId("lf_foot"), xref, pinocchio.LOCAL, gains
    )
    CONTACT_DER = Contact1DModelDerived(
        ROBOT_STATE, ROBOT_MODEL.getFrameId("lf_foot"), xref, pinocchio.LOCAL, gains
    )
    Raxis = pinocchio.SE3.Random().rotation
    CONTACT.Raxis = Raxis
    CONTACT_DER.Raxis = Raxis


class Contact1DWorldTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("hyq").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    xref = pinocchio.SE3.Random().translation[2]
    CONTACT = crocoddyl.ContactModel1D(
        ROBOT_STATE, ROBOT_MODEL.getFrameId("lf_foot"), xref, pinocchio.WORLD, gains
    )
    CONTACT_DER = Contact1DModelDerived(
        ROBOT_STATE, ROBOT_MODEL.getFrameId("lf_foot"), xref, pinocchio.WORLD, gains
    )
    Raxis = pinocchio.SE3.Random().rotation
    CONTACT.Raxis = Raxis
    CONTACT_DER.Raxis = Raxis


class Contact1DLocalWorldAlignedTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("hyq").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    xref = pinocchio.SE3.Random().translation[2]
    CONTACT = crocoddyl.ContactModel1D(
        ROBOT_STATE,
        ROBOT_MODEL.getFrameId("lf_foot"),
        xref,
        pinocchio.LOCAL_WORLD_ALIGNED,
        gains,
    )
    CONTACT_DER = Contact1DModelDerived(
        ROBOT_STATE,
        ROBOT_MODEL.getFrameId("lf_foot"),
        xref,
        pinocchio.LOCAL_WORLD_ALIGNED,
        gains,
    )
    Raxis = pinocchio.SE3.Random().rotation
    CONTACT.Raxis = Raxis
    CONTACT_DER.Raxis = Raxis


class Contact3DLocalTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("hyq").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    xref = pinocchio.SE3.Random().translation
    CONTACT = crocoddyl.ContactModel3D(
        ROBOT_STATE, ROBOT_MODEL.getFrameId("lf_foot"), xref, pinocchio.LOCAL, gains
    )
    CONTACT_DER = Contact3DModelDerived(
        ROBOT_STATE, ROBOT_MODEL.getFrameId("lf_foot"), xref, pinocchio.LOCAL, gains
    )


class Contact3DWorldTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("hyq").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    xref = pinocchio.SE3.Random().translation
    CONTACT = crocoddyl.ContactModel3D(
        ROBOT_STATE, ROBOT_MODEL.getFrameId("lf_foot"), xref, pinocchio.WORLD, gains
    )
    CONTACT_DER = Contact3DModelDerived(
        ROBOT_STATE, ROBOT_MODEL.getFrameId("lf_foot"), xref, pinocchio.WORLD, gains
    )


class Contact3DLocalWorldAlignedTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("hyq").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    xref = pinocchio.SE3.Random().translation
    CONTACT = crocoddyl.ContactModel3D(
        ROBOT_STATE,
        ROBOT_MODEL.getFrameId("lf_foot"),
        xref,
        pinocchio.LOCAL_WORLD_ALIGNED,
        gains,
    )
    CONTACT_DER = Contact3DModelDerived(
        ROBOT_STATE,
        ROBOT_MODEL.getFrameId("lf_foot"),
        xref,
        pinocchio.LOCAL_WORLD_ALIGNED,
        gains,
    )


class Contact3DMultipleTest(ContactModelMultipleAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("hyq").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    CONTACTS = collections.OrderedDict(
        sorted(
            {
                "lf_foot": crocoddyl.ContactModel3D(
                    ROBOT_STATE,
                    ROBOT_MODEL.getFrameId("lf_foot"),
                    pinocchio.SE3.Random().translation,
                    pinocchio.LOCAL,
                    gains,
                ),
                "rh_foot": crocoddyl.ContactModel3D(
                    ROBOT_STATE,
                    ROBOT_MODEL.getFrameId("rh_foot"),
                    pinocchio.SE3.Random().translation,
                    pinocchio.LOCAL,
                    gains,
                ),
            }.items()
        )
    )


class Contact6DLocalTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    Mref = pinocchio.SE3.Random()
    CONTACT = crocoddyl.ContactModel6D(
        ROBOT_STATE, ROBOT_MODEL.getFrameId("r_sole"), Mref, pinocchio.LOCAL, gains
    )
    CONTACT_DER = Contact6DModelDerived(
        ROBOT_STATE, ROBOT_MODEL.getFrameId("r_sole"), Mref, pinocchio.LOCAL, gains
    )


class Contact6DWorldTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    Mref = pinocchio.SE3.Random()
    CONTACT = crocoddyl.ContactModel6D(
        ROBOT_STATE, ROBOT_MODEL.getFrameId("r_sole"), Mref, pinocchio.WORLD, gains
    )
    CONTACT_DER = Contact6DModelDerived(
        ROBOT_STATE, ROBOT_MODEL.getFrameId("r_sole"), Mref, pinocchio.WORLD, gains
    )


class Contact6DLocalWorldAlignedTest(ContactModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    Mref = pinocchio.SE3.Random()
    CONTACT = crocoddyl.ContactModel6D(
        ROBOT_STATE,
        ROBOT_MODEL.getFrameId("r_sole"),
        Mref,
        pinocchio.LOCAL_WORLD_ALIGNED,
        gains,
    )
    CONTACT_DER = Contact6DModelDerived(
        ROBOT_STATE,
        ROBOT_MODEL.getFrameId("r_sole"),
        Mref,
        pinocchio.LOCAL_WORLD_ALIGNED,
        gains,
    )


class Contact6DMultipleTest(ContactModelMultipleAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    gains = pinocchio.utils.rand(2)
    CONTACTS = collections.OrderedDict(
        sorted(
            {
                "l_foot": crocoddyl.ContactModel6D(
                    ROBOT_STATE,
                    ROBOT_MODEL.getFrameId("l_sole"),
                    pinocchio.SE3.Random(),
                    pinocchio.LOCAL,
                    gains,
                ),
                "r_foot": crocoddyl.ContactModel6D(
                    ROBOT_STATE,
                    ROBOT_MODEL.getFrameId("r_sole"),
                    pinocchio.SE3.Random(),
                    pinocchio.LOCAL,
                    gains,
                ),
            }.items()
        )
    )


if __name__ == "__main__":
    # test to be run
    test_classes_to_run = [
        Contact1DLocalTest,
        Contact1DWorldTest,
        Contact1DLocalWorldAlignedTest,
        Contact3DLocalTest,
        Contact3DWorldTest,
        Contact3DLocalWorldAlignedTest,
        Contact3DMultipleTest,
        Contact6DLocalTest,
        Contact6DWorldTest,
        Contact6DLocalWorldAlignedTest,
        Contact6DMultipleTest,
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
