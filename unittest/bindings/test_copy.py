import copy
import sys
import unittest

import numpy as np
import pinocchio

import crocoddyl


class CopyModelTestCase(unittest.TestCase):
    MODEL = list()
    DATA = False
    COLLECTOR = list()

    def test_copy(self):
        Mcopy = copy.copy(self.MODEL)
        self.assertFalse(id(self.MODEL) == id(Mcopy))
        if self.DATA:
            D = []
            for i, m in enumerate(self.MODEL):
                if not self.COLLECTOR:
                    D.append(m.createData())
                else:
                    D.append(m.createData(self.COLLECTOR[i]))
            Dcopy = copy.copy(D)
            self.assertFalse(id(D) == id(Dcopy))
        for i, m in enumerate(self.MODEL):
            self.assertTrue(id(self.MODEL[i]) == id(Mcopy[i]))
            if self.DATA:
                self.assertTrue(id(D[i]) == id(Dcopy[i]))

    def test_deepcopy(self):
        Mcopy = copy.deepcopy(self.MODEL)
        self.assertFalse(id(self.MODEL) == id(Mcopy))
        if self.DATA:
            D = []
            for i, m in enumerate(self.MODEL):
                if not self.COLLECTOR:
                    D.append(m.createData())
                else:
                    D.append(m.createData(self.COLLECTOR[i]))
            Dcopy = copy.deepcopy(D)
            self.assertFalse(id(D) == id(Dcopy))
        for i, m in enumerate(self.MODEL):
            self.assertFalse(id(self.MODEL[i]) == id(Mcopy[i]))
            if self.DATA:
                self.assertFalse(id(D[i]) == id(Dcopy[i]))


class ActionsTest(CopyModelTestCase):
    MODEL = list()
    DATA = True
    # core actions
    MODEL.append(crocoddyl.ActionModelUnicycle())
    MODEL.append(crocoddyl.ActionModelLQR(2, 2))
    MODEL.append(crocoddyl.DifferentialActionModelLQR(2, 2))
    # multibody actions
    state = crocoddyl.StateMultibody(pinocchio.buildSampleModelHumanoidRandom())
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    cost = crocoddyl.CostModelSum(state, actuation.nu)
    impulse = crocoddyl.ImpulseModelMultiple(state)
    contact = crocoddyl.ContactModelMultiple(state, actuation.nu)
    MODEL.append(crocoddyl.ActionModelImpulseFwdDynamics(state, impulse, cost))
    MODEL.append(
        crocoddyl.DifferentialActionModelContactFwdDynamics(
            state, actuation, contact, cost
        )
    )
    MODEL.append(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, cost)
    )
    cost = crocoddyl.CostModelSum(state, state.nv)
    contact = crocoddyl.ContactModelMultiple(state, state.nv)
    MODEL.append(
        crocoddyl.DifferentialActionModelContactInvDynamics(
            state, actuation, contact, cost
        )
    )
    MODEL.append(
        crocoddyl.DifferentialActionModelFreeInvDynamics(state, actuation, cost)
    )
    # integrated actions
    MODEL.append(
        crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelLQR(2, 2), 0.1
        )
    )
    MODEL.append(
        crocoddyl.IntegratedActionModelRK(
            crocoddyl.DifferentialActionModelLQR(2, 2), crocoddyl.RKType.two, 0.1
        )
    )
    # numdiff actions
    MODEL.append(crocoddyl.ActionModelNumDiff(crocoddyl.ActionModelLQR(2, 2)))
    MODEL.append(
        crocoddyl.DifferentialActionModelNumDiff(
            crocoddyl.DifferentialActionModelLQR(2, 2)
        )
    )


class StatesTest(CopyModelTestCase):
    MODEL = list()
    # core states
    MODEL.append(crocoddyl.StateVector(2))
    MODEL.append(crocoddyl.StateNumDiff(crocoddyl.StateVector(2)))
    # multibody states
    MODEL.append(crocoddyl.StateMultibody(pinocchio.buildSampleModelHumanoidRandom()))


class ResidualsTest(CopyModelTestCase):
    MODEL = list()
    DATA = True
    COLLECTOR = list()
    # core residuals
    state = crocoddyl.StateMultibody(pinocchio.buildSampleModelHumanoidRandom())
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    joint = crocoddyl.JointDataAbstract(state, actuation, actuation.nu)
    MODEL.append(crocoddyl.ResidualModelControl(crocoddyl.StateVector(2)))
    COLLECTOR.append(crocoddyl.DataCollectorJoint(joint))
    MODEL.append(crocoddyl.ResidualModelJointEffort(state, actuation))
    COLLECTOR.append(crocoddyl.DataCollectorJoint(joint))
    MODEL.append(crocoddyl.ResidualModelJointAcceleration(state))
    COLLECTOR.append(crocoddyl.DataCollectorJoint(joint))
    # multibody residuals
    # TODO(cmastalli): add pair-collision residual
    frame_id = state.pinocchio.getFrameId("rleg6_joint")
    contact = crocoddyl.ContactModelMultiple(state, actuation.nu)
    contact.addContact(
        "rleg6_contact",
        crocoddyl.ContactModel6D(
            state,
            frame_id,
            pinocchio.SE3.Random(),
            pinocchio.LOCAL,
            actuation.nu,
            np.zeros(2),
        ),
    )
    pdata = state.pinocchio.createData()
    adata = actuation.createData()
    cdata = contact.createData(pdata)
    MODEL.append(crocoddyl.ResidualModelCentroidalMomentum(state, np.zeros(6)))
    COLLECTOR.append(crocoddyl.DataCollectorMultibody(pdata))
    MODEL.append(crocoddyl.ResidualModelCoMPosition(state, np.zeros(3)))
    COLLECTOR.append(crocoddyl.DataCollectorMultibody(pdata))
    MODEL.append(crocoddyl.ResidualModelControlGrav(state))
    COLLECTOR.append(crocoddyl.DataCollectorActMultibody(pdata, adata))
    MODEL.append(
        crocoddyl.ResidualModelFramePlacement(state, frame_id, pinocchio.SE3.Random())
    )
    COLLECTOR.append(crocoddyl.DataCollectorMultibody(pdata))
    MODEL.append(
        crocoddyl.ResidualModelFrameRotation(
            state, frame_id, pinocchio.SE3.Random().rotation
        )
    )
    COLLECTOR.append(crocoddyl.DataCollectorMultibody(pdata))
    MODEL.append(
        crocoddyl.ResidualModelFrameTranslation(
            state, frame_id, pinocchio.SE3.Random().translation
        )
    )
    COLLECTOR.append(crocoddyl.DataCollectorMultibody(pdata))
    MODEL.append(
        crocoddyl.ResidualModelFrameVelocity(
            state, frame_id, pinocchio.Motion.Random(), pinocchio.ReferenceFrame.LOCAL
        )
    )
    COLLECTOR.append(crocoddyl.DataCollectorMultibody(pdata))
    MODEL.append(crocoddyl.ResidualModelState(state))
    COLLECTOR.append(crocoddyl.DataCollectorAbstract())
    MODEL.append(crocoddyl.ResidualModelContactControlGrav(state))
    COLLECTOR.append(crocoddyl.DataCollectorActMultibodyInContact(pdata, adata, cdata))
    MODEL.append(
        crocoddyl.ResidualModelContactCoPPosition(
            state, frame_id, crocoddyl.CoPSupport()
        )
    )
    COLLECTOR.append(crocoddyl.DataCollectorActMultibodyInContact(pdata, adata, cdata))
    MODEL.append(
        crocoddyl.ResidualModelContactForce(
            state, frame_id, pinocchio.Force.Random(), 6
        )
    )
    COLLECTOR.append(crocoddyl.DataCollectorActMultibodyInContact(pdata, adata, cdata))
    MODEL.append(
        crocoddyl.ResidualModelContactFrictionCone(
            state, frame_id, crocoddyl.FrictionCone()
        )
    )
    COLLECTOR.append(crocoddyl.DataCollectorActMultibodyInContact(pdata, adata, cdata))
    MODEL.append(
        crocoddyl.ResidualModelContactWrenchCone(
            state, frame_id, crocoddyl.WrenchCone()
        )
    )
    COLLECTOR.append(crocoddyl.DataCollectorActMultibodyInContact(pdata, adata, cdata))


class ActivationsTest(CopyModelTestCase):
    MODEL = list()
    DATA = True
    bounds = crocoddyl.ActivationBounds(np.zeros(2), np.ones(2), 0.1)
    MODEL.append(crocoddyl.ActivationModel2NormBarrier(2))
    MODEL.append(crocoddyl.ActivationModelQuadraticBarrier(bounds))
    MODEL.append(crocoddyl.ActivationModelQuadFlatExp(2, 0.1))
    MODEL.append(crocoddyl.ActivationModelQuadFlatLog(2, 0.1))
    MODEL.append(crocoddyl.ActivationModelQuad(2))
    MODEL.append(crocoddyl.ActivationModelSmooth1Norm(2, 0.1))
    MODEL.append(crocoddyl.ActivationModelSmooth2Norm(2, 0.1))
    MODEL.append(crocoddyl.ActivationModelWeightedQuadraticBarrier(bounds, np.ones(2)))
    MODEL.append(crocoddyl.ActivationModelWeightedQuad(np.ones(2)))
    MODEL.append(
        crocoddyl.ActivationModelNumDiff(crocoddyl.ActivationModel2NormBarrier(2))
    )


class CostsTest(CopyModelTestCase):
    MODEL = list()
    DATA = True
    COLLECTOR = list()
    state = crocoddyl.StateVector(2)
    residual = crocoddyl.ResidualModelControl(crocoddyl.StateVector(2))
    activation = crocoddyl.ActivationModelWeightedQuad(np.ones(1))
    MODEL.append(crocoddyl.CostModelResidual(state, activation, residual))
    COLLECTOR.append(crocoddyl.DataCollectorAbstract())
    MODEL.append(crocoddyl.CostModelSum(state, 2))
    COLLECTOR.append(crocoddyl.DataCollectorAbstract())


class ConstraintsTest(CopyModelTestCase):
    MODEL = list()
    DATA = True
    COLLECTOR = list()
    state = crocoddyl.StateVector(2)
    residual = crocoddyl.ResidualModelControl(crocoddyl.StateVector(2))
    MODEL.append(crocoddyl.ConstraintModelResidual(state, residual))
    COLLECTOR.append(crocoddyl.DataCollectorAbstract())
    MODEL.append(crocoddyl.ConstraintModelManager(state, 2))
    COLLECTOR.append(crocoddyl.DataCollectorAbstract())


class ControlsTest(CopyModelTestCase):
    MODEL = list()
    DATA = True
    MODEL.append(crocoddyl.ControlParametrizationModelPolyZero(2))
    MODEL.append(crocoddyl.ControlParametrizationModelPolyOne(2))
    MODEL.append(
        crocoddyl.ControlParametrizationModelPolyTwoRK(2, crocoddyl.RKType.three)
    )


class DataCollectorsTest(CopyModelTestCase):
    MODEL = list()
    DATA = False
    # core collectors
    state = crocoddyl.StateMultibody(pinocchio.buildSampleModelHumanoidRandom())
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    jdata = crocoddyl.JointDataAbstract(state, actuation, actuation.nu)
    MODEL.append(crocoddyl.DataCollectorAbstract())
    MODEL.append(crocoddyl.DataCollectorActuation(actuation.createData()))
    MODEL.append(jdata)
    MODEL.append(crocoddyl.DataCollectorJoint(jdata))
    # multibody collectors
    impulse = crocoddyl.ImpulseModelMultiple(state)
    contact = crocoddyl.ContactModelMultiple(state, actuation.nu)
    pdata = state.pinocchio.createData()
    adata = actuation.createData()
    cdata = contact.createData(pdata)
    idata = impulse.createData(pdata)
    MODEL.append(crocoddyl.DataCollectorMultibody(pdata))
    MODEL.append(crocoddyl.DataCollectorActMultibody(pdata, adata))
    MODEL.append(crocoddyl.DataCollectorJointActMultibody(pdata, adata, jdata))
    MODEL.append(crocoddyl.DataCollectorImpulse(idata))
    MODEL.append(crocoddyl.DataCollectorContact(cdata))
    MODEL.append(crocoddyl.DataCollectorMultibodyInImpulse(pdata, idata))
    MODEL.append(crocoddyl.DataCollectorMultibodyInContact(pdata, cdata))
    MODEL.append(crocoddyl.DataCollectorActMultibodyInContact(pdata, adata, cdata))
    MODEL.append(
        crocoddyl.DataCollectorJointActMultibodyInContact(pdata, adata, jdata, cdata)
    )
    cmodel = crocoddyl.ContactModelAbstract(state, pinocchio.LOCAL, 3, actuation.nu)
    MODEL.append(crocoddyl.ForceDataAbstract(cmodel, pdata))


class ActuationsTest(CopyModelTestCase):
    MODEL = list()
    DATA = True
    # core actuations
    state = crocoddyl.StateMultibody(pinocchio.buildSampleModelHumanoidRandom())
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    MODEL.append(crocoddyl.SquashingModelSmoothSat(np.zeros(2), np.ones(2), 2))
    MODEL.append(
        crocoddyl.ActuationSquashingModel(
            actuation,
            crocoddyl.SquashingModelSmoothSat(np.zeros(2), np.ones(2), 2),
            actuation.nu,
        )
    )
    # multibody actuations
    MODEL.append(crocoddyl.ActuationModelFloatingBase(state))
    MODEL.append(crocoddyl.ActuationModelFull(state))
    d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
    ps = [
        crocoddyl.Thruster(
            pinocchio.SE3(np.eye(3), np.array([d_cog, 0, 0])),
            cm / cf,
            crocoddyl.ThrusterType.CCW,
        ),
        crocoddyl.Thruster(
            pinocchio.SE3(np.eye(3), np.array([0, d_cog, 0])),
            cm / cf,
            crocoddyl.ThrusterType.CW,
        ),
        crocoddyl.Thruster(
            pinocchio.SE3(np.eye(3), np.array([-d_cog, 0, 0])),
            cm / cf,
            crocoddyl.ThrusterType.CCW,
        ),
        crocoddyl.Thruster(
            pinocchio.SE3(np.eye(3), np.array([0, -d_cog, 0])),
            cm / cf,
            crocoddyl.ThrusterType.CW,
        ),
    ]
    actuation = crocoddyl.ActuationModelFloatingBaseThrusters(state, ps)
    MODEL.append(crocoddyl.ActuationModelFloatingBaseThrusters(state, ps))


class ContactsTest(CopyModelTestCase):
    MODEL = list()
    DATA = True
    COLLECTOR = list()
    state = crocoddyl.StateMultibody(pinocchio.buildSampleModelHumanoidRandom())
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    frame_id = state.pinocchio.getFrameId("rleg6_joint")
    pdata = state.pinocchio.createData()
    # contact models
    MODEL.append(crocoddyl.ContactModelMultiple(state, actuation.nu))
    COLLECTOR.append(pdata)
    MODEL.append(
        crocoddyl.ContactModel1D(
            state, frame_id, 0.0, pinocchio.LOCAL, np.eye(3), actuation.nu, np.zeros(2)
        )
    )
    COLLECTOR.append(pdata)
    MODEL.append(
        crocoddyl.ContactModel2D(state, frame_id, np.ones(2), actuation.nu, np.zeros(2))
    )
    COLLECTOR.append(pdata)
    MODEL.append(
        crocoddyl.ContactModel3D(
            state, frame_id, np.ones(3), pinocchio.LOCAL, actuation.nu, np.zeros(2)
        )
    )
    COLLECTOR.append(pdata)
    MODEL.append(
        crocoddyl.ContactModel6D(
            state,
            frame_id,
            pinocchio.SE3.Random(),
            pinocchio.LOCAL,
            actuation.nu,
            np.zeros(2),
        )
    )
    COLLECTOR.append(pdata)
    # impulse models
    MODEL.append(crocoddyl.ImpulseModelMultiple(state))
    COLLECTOR.append(pdata)
    MODEL.append(crocoddyl.ImpulseModel3D(state, frame_id, pinocchio.LOCAL))
    COLLECTOR.append(pdata)
    MODEL.append(crocoddyl.ImpulseModel6D(state, frame_id))
    COLLECTOR.append(pdata)


class ConesTest(CopyModelTestCase):
    MODEL = list()
    MODEL.append(crocoddyl.FrictionCone())
    MODEL.append(crocoddyl.WrenchCone())
    MODEL.append(crocoddyl.CoPSupport())


class ProblemAndSolversTest(CopyModelTestCase):
    MODEL = list()
    m = crocoddyl.ActionModelLQR(2, 2)
    problem = crocoddyl.ShootingProblem(m.state.zero(), [m] * 10, m)
    MODEL.append(problem)
    MODEL.append(crocoddyl.CallbackVerbose())
    MODEL.append(crocoddyl.SolverKKT(problem))
    MODEL.append(crocoddyl.SolverDDP(problem))
    MODEL.append(crocoddyl.SolverFDDP(problem))
    MODEL.append(crocoddyl.SolverBoxDDP(problem))
    MODEL.append(crocoddyl.SolverBoxFDDP(problem))
    MODEL.append(crocoddyl.SolverIntro(problem))
    if hasattr(crocoddyl, "SolverIpopt"):
        MODEL.append(crocoddyl.SolverIpopt(problem))


if __name__ == "__main__":
    # test to be run
    test_classes_to_run = [
        ActionsTest,
        StatesTest,
        ResidualsTest,
        ActivationsTest,
        CostsTest,
        ConstraintsTest,
        ControlsTest,
        DataCollectorsTest,
        ActuationsTest,
        ContactsTest,
        ConesTest,
        ProblemAndSolversTest,
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
