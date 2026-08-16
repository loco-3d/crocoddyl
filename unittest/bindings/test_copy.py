###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################

import copy
import sys
import unittest

import numpy as np
import pinocchio

import crocoddyl


class CopyModelTestCase(unittest.TestCase):
    MODEL = []
    DATA = False
    COLLECTOR = []

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
    MODEL = []
    DATA = True
    # core actions
    MODEL.append(crocoddyl.ActionModelUnicycle())
    MODEL.append(crocoddyl.ActionModelLQR(2, 2))
    MODEL.append(crocoddyl.DifferentialActionModelLQR(2, 2))
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


class DynamicsModelsTest(CopyModelTestCase):
    MODEL = []
    DATA = True
    state = crocoddyl.StateMultibody(pinocchio.buildSampleModelManipulator())
    actuation = crocoddyl.ActuationModelMultibody(state)
    forward_constraints = crocoddyl.ImplicitConstraintModelMultiple(state, actuation.nu)
    inverse_constraints = crocoddyl.ImplicitConstraintModelMultiple(state, state.nv)
    impulse_constraints = crocoddyl.ImplicitConstraintModelMultiple(state, 0)
    MODEL.append(
        crocoddyl.DynamicsModelConstrainedForward(state, actuation, forward_constraints)
    )
    MODEL.append(
        crocoddyl.DynamicsModelConstrainedInverse(state, actuation, inverse_constraints)
    )
    MODEL.append(crocoddyl.DynamicsModelImpulseForward(state, impulse_constraints))


class StatesTest(CopyModelTestCase):
    MODEL = []
    # core states
    MODEL.append(crocoddyl.StateVector(2))
    MODEL.append(crocoddyl.StateNumDiff(crocoddyl.StateVector(2)))
    # multibody states
    MODEL.append(crocoddyl.StateMultibody(pinocchio.buildSampleModelHumanoidRandom()))


class ResidualsTest(CopyModelTestCase):
    MODEL = []
    DATA = True
    COLLECTOR = []
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
    contact = crocoddyl.ContactModel(
        state,
        frame_id,
        pinocchio.SE3.Random(),
        pinocchio.LOCAL,
        actuation.nu,
        np.zeros(2),
        [True] * 6,
    )
    constraints = crocoddyl.ImplicitConstraintModelMultiple(state, actuation.nu)
    constraints.addConstraint("rleg6_contact", contact)
    pdata = state.pinocchio.createData()
    adata = actuation.createData()
    cdata = constraints.createData(pdata)
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
    COLLECTOR.append(
        crocoddyl.DataCollectorActMultibodyInImplicitConstraint(pdata, adata, cdata)
    )
    MODEL.append(
        crocoddyl.ResidualModelContactCoPPosition(
            state, frame_id, crocoddyl.CoPSupport()
        )
    )
    COLLECTOR.append(
        crocoddyl.DataCollectorActMultibodyInImplicitConstraint(pdata, adata, cdata)
    )
    MODEL.append(
        crocoddyl.ResidualModelContactForce(
            state, frame_id, pinocchio.Force.Random(), 6
        )
    )
    COLLECTOR.append(
        crocoddyl.DataCollectorActMultibodyInImplicitConstraint(pdata, adata, cdata)
    )
    MODEL.append(
        crocoddyl.ResidualModelContactFrictionCone(
            state, frame_id, crocoddyl.FrictionCone()
        )
    )
    COLLECTOR.append(
        crocoddyl.DataCollectorActMultibodyInImplicitConstraint(pdata, adata, cdata)
    )
    MODEL.append(
        crocoddyl.ResidualModelContactWrenchCone(
            state, frame_id, crocoddyl.WrenchCone()
        )
    )
    COLLECTOR.append(
        crocoddyl.DataCollectorActMultibodyInImplicitConstraint(pdata, adata, cdata)
    )


class ActivationsTest(CopyModelTestCase):
    MODEL = []
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
    MODEL = []
    DATA = True
    COLLECTOR = []
    state = crocoddyl.StateVector(2)
    residual = crocoddyl.ResidualModelControl(crocoddyl.StateVector(2))
    activation = crocoddyl.ActivationModelWeightedQuad(np.ones(1))
    MODEL.append(crocoddyl.CostModelResidual(state, activation, residual))
    COLLECTOR.append(crocoddyl.DataCollectorAbstract())
    MODEL.append(crocoddyl.CostModelSum(state, 2))
    COLLECTOR.append(crocoddyl.DataCollectorAbstract())


class ConstraintsTest(CopyModelTestCase):
    MODEL = []
    DATA = True
    COLLECTOR = []
    state = crocoddyl.StateVector(2)
    residual = crocoddyl.ResidualModelControl(crocoddyl.StateVector(2))
    MODEL.append(crocoddyl.ConstraintModelResidual(state, residual))
    COLLECTOR.append(crocoddyl.DataCollectorAbstract())
    MODEL.append(crocoddyl.ConstraintModelManager(state, 2))
    COLLECTOR.append(crocoddyl.DataCollectorAbstract())


class ControlsTest(CopyModelTestCase):
    MODEL = []
    DATA = True
    MODEL.append(crocoddyl.ControlParametrizationModelPolyZero(2))
    MODEL.append(crocoddyl.ControlParametrizationModelPolyOne(2))
    MODEL.append(
        crocoddyl.ControlParametrizationModelPolyTwoRK(2, crocoddyl.RKType.three)
    )


class JointDynamicsTest(CopyModelTestCase):
    MODEL = []
    DATA = True
    if hasattr(crocoddyl, "JointDynamicsModelIdentity"):
        MODEL.append(crocoddyl.JointDynamicsModelIdentity(2, 1, 1))
    if hasattr(crocoddyl, "JointDynamicsModelFriction"):
        MODEL.append(
            crocoddyl.JointDynamicsModelFriction(
                3,
                1,
                np.array([np.log(0.2), np.log(4.0), np.log(0.1)]),
                crocoddyl.JointFrictionType.COULOMB_VISCOUS,
            )
        )
    if hasattr(crocoddyl, "JointDynamicsModelThruster"):
        MODEL.append(
            crocoddyl.JointDynamicsModelThruster(
                [
                    crocoddyl.Thruster(
                        pinocchio.SE3.Identity(),
                        0.1,
                        crocoddyl.ThrusterType.CW,
                    )
                ]
            )
        )


class DataCollectorsTest(CopyModelTestCase):
    MODEL = []
    DATA = False
    # core collectors
    state = crocoddyl.StateMultibody(pinocchio.buildSampleModelHumanoidRandom())
    actuation = crocoddyl.ActuationModelFloatingBase(state)
    jdata = crocoddyl.JointDataAbstract(state, actuation, actuation.nu)
    params = crocoddyl.ParamsDataAbstract(2, 3)
    MODEL.append(params)
    MODEL.append(crocoddyl.DataCollectorAbstract())
    MODEL.append(crocoddyl.DataCollectorParams(params))
    MODEL.append(crocoddyl.DataCollectorActuation(actuation.createData()))
    MODEL.append(crocoddyl.DataCollectorActuationParams(actuation.createData(), params))
    MODEL.append(jdata)
    MODEL.append(crocoddyl.DataCollectorJoint(jdata))
    MODEL.append(crocoddyl.DataCollectorJointParams(jdata, params))
    MODEL.append(
        crocoddyl.DataCollectorJointActuationParams(
            actuation.createData(), jdata, params
        )
    )
    # multibody collectors
    constraints = crocoddyl.ImplicitConstraintModelMultiple(state, actuation.nu)
    pdata = state.pinocchio.createData()
    adata = actuation.createData()
    cdata = constraints.createData(pdata)
    MODEL.append(crocoddyl.DataCollectorMultibody(pdata))
    MODEL.append(crocoddyl.DataCollectorMultibodyParams(pdata, params))
    MODEL.append(crocoddyl.DataCollectorActMultibody(pdata, adata))
    MODEL.append(crocoddyl.DataCollectorActMultibodyParams(pdata, adata, params))
    MODEL.append(crocoddyl.DataCollectorJointActMultibody(pdata, adata, jdata))
    MODEL.append(
        crocoddyl.DataCollectorJointActMultibodyParams(pdata, adata, jdata, params)
    )
    MODEL.append(crocoddyl.DataCollectorImplicitConstraint(cdata))
    MODEL.append(crocoddyl.DataCollectorMultibodyInImplicitConstraint(pdata, cdata))
    MODEL.append(
        crocoddyl.DataCollectorMultibodyInImplicitConstraintParams(pdata, cdata, params)
    )
    MODEL.append(
        crocoddyl.DataCollectorActMultibodyInImplicitConstraint(pdata, adata, cdata)
    )
    MODEL.append(
        crocoddyl.DataCollectorActMultibodyInImplicitConstraintParams(
            pdata, adata, cdata, params
        )
    )
    MODEL.append(
        crocoddyl.DataCollectorJointActMultibodyInImplicitConstraint(
            pdata, adata, jdata, cdata
        )
    )
    MODEL.append(
        crocoddyl.DataCollectorJointActMultibodyInImplicitConstraintParams(
            pdata, adata, jdata, cdata, params
        )
    )


class ActuationsTest(CopyModelTestCase):
    MODEL = []
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
    if hasattr(crocoddyl, "ActuationModelMultibody"):
        MODEL.append(crocoddyl.ActuationModelMultibody(state))
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
    if hasattr(crocoddyl, "ActuationModelMultibody") and hasattr(
        crocoddyl, "JointDynamicsModelThruster"
    ):
        MODEL.append(
            crocoddyl.ActuationModelMultibody(
                state, [crocoddyl.JointDynamicsModelThruster(ps)]
            )
        )


class ConesTest(CopyModelTestCase):
    MODEL = []
    MODEL.append(crocoddyl.FrictionCone())
    MODEL.append(crocoddyl.WrenchCone())
    MODEL.append(crocoddyl.CoPSupport())


class ProblemAndSolversTest(CopyModelTestCase):
    MODEL = []
    m = crocoddyl.ActionModelLQR(2, 2)
    problem = crocoddyl.ShootingProblem(m.state.zero(), [m] * 10, m)
    MODEL.append(problem)
    MODEL.append(crocoddyl.CallbackVerbose())
    MODEL.append(crocoddyl.SolverKKT(problem))
    MODEL.append(crocoddyl.SolverFDDP(problem))
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
        JointDynamicsTest,
        DataCollectorsTest,
        ActuationsTest,
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
