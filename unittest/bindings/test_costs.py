import sys
import unittest

import example_robot_data
import numpy as np

import crocoddyl
import pinocchio
from crocoddyl.utils import (CoMPositionCostModelDerived, ControlCostModelDerived, FramePlacementCostModelDerived,
                             FrameRotationCostModelDerived, FrameTranslationCostModelDerived,
                             FrameVelocityCostModelDerived, StateCostModelDerived)


class CostModelAbstractTestCase(unittest.TestCase):
    ROBOT_MODEL = None
    ROBOT_STATE = None
    COST = None
    COST_DER = None

    def setUp(self):
        self.robot_data = self.ROBOT_MODEL.createData()
        self.x = self.ROBOT_STATE.rand()
        self.u = pinocchio.utils.rand(self.ROBOT_MODEL.nv)

        self.multibody_data = crocoddyl.DataCollectorMultibody(self.robot_data)
        self.data = self.COST.createData(self.multibody_data)
        self.data_der = self.COST_DER.createData(self.multibody_data)

        nq, nv = self.ROBOT_MODEL.nq, self.ROBOT_MODEL.nv
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:])
        pinocchio.computeForwardKinematicsDerivatives(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:],
                                                      pinocchio.utils.zero(nv))
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data, self.x[:nq])
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.jacobianCenterOfMass(self.ROBOT_MODEL, self.robot_data, self.x[:nq], False)

    def test_dimensions(self):
        self.assertEqual(self.COST.state.nx, self.COST_DER.state.nx, "Wrong nx.")
        self.assertEqual(self.COST.state.ndx, self.COST_DER.state.ndx, "Wrong ndx.")
        self.assertEqual(self.COST.nu, self.COST_DER.nu, "Wrong nu.")
        self.assertEqual(self.COST.state.nq, self.COST_DER.state.nq, "Wrong nq.")
        self.assertEqual(self.COST.state.nv, self.COST_DER.state.nv, "Wrong nv.")
        self.assertEqual(self.COST.activation.nr, self.COST_DER.activation.nr, "Wrong nr.")

    def test_calc(self):
        # Run calc for both action models
        self.COST.calc(self.data, self.x, self.u)
        self.COST_DER.calc(self.data_der, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.data.cost, self.data_der.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.data.residual.r, self.data_der.residual.r, atol=1e-9),
                        "Wrong cost residuals.")

    def test_calcDiff(self):
        # Run calc for both action models
        self.COST.calc(self.data, self.x, self.u)
        self.COST.calcDiff(self.data, self.x, self.u)

        self.COST_DER.calc(self.data_der, self.x, self.u)
        self.COST_DER.calcDiff(self.data_der, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.data.cost, self.data_der.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.data.residual.r, self.data_der.residual.r, atol=1e-9),
                        "Wrong cost residuals.")
        # Checking the Jacobians and Hessians of the cost
        self.assertTrue(np.allclose(self.data.Lx, self.data_der.Lx, atol=1e-9), "Wrong Lx.")
        self.assertTrue(np.allclose(self.data.Lu, self.data_der.Lu, atol=1e-9), "Wrong Lu.")
        self.assertTrue(np.allclose(self.data.Lxx, self.data_der.Lxx, atol=1e-9), "Wrong Lxx.")
        self.assertTrue(np.allclose(self.data.Lxu, self.data_der.Lxu, atol=1e-9), "Wrong Lxu.")
        self.assertTrue(np.allclose(self.data.Luu, self.data_der.Luu, atol=1e-9), "Wrong Luu.")


class CostModelSumTestCase(unittest.TestCase):
    ROBOT_MODEL = None
    ROBOT_STATE = None
    COST = None

    def setUp(self):
        self.robot_data = self.ROBOT_MODEL.createData()
        self.x = self.ROBOT_STATE.rand()
        self.u = pinocchio.utils.rand(self.ROBOT_MODEL.nv)

        self.cost_sum = crocoddyl.CostModelSum(self.ROBOT_STATE)
        self.cost_sum.addCost('myCost', self.COST, 1.)

        self.multibody_data = crocoddyl.DataCollectorMultibody(self.robot_data)
        self.data = self.COST.createData(self.multibody_data)
        self.data_sum = self.cost_sum.createData(self.multibody_data)

        nq, nv = self.ROBOT_MODEL.nq, self.ROBOT_MODEL.nv
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:])
        pinocchio.computeForwardKinematicsDerivatives(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:],
                                                      pinocchio.utils.zero(nv))
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data, self.x[:nq])
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.jacobianCenterOfMass(self.ROBOT_MODEL, self.robot_data, self.x[:nq], False)

    def test_dimensions(self):
        self.assertEqual(self.COST.state.nx, self.cost_sum.state.nx, "Wrong nx.")
        self.assertEqual(self.COST.state.ndx, self.cost_sum.state.ndx, "Wrong ndx.")
        self.assertEqual(self.COST.nu, self.cost_sum.nu, "Wrong nu.")
        self.assertEqual(self.COST.state.nq, self.cost_sum.state.nq, "Wrong nq.")
        self.assertEqual(self.COST.state.nv, self.cost_sum.state.nv, "Wrong nv.")
        self.assertEqual(self.COST.activation.nr, self.cost_sum.nr, "Wrong nr.")

    def test_calc(self):
        # Run calc for both action models
        self.COST.calc(self.data, self.x, self.u)
        self.cost_sum.calc(self.data_sum, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.data.cost, self.data_sum.cost, 10, "Wrong cost value.")

    def test_calcDiff(self):
        # Run calc for both action models
        self.COST.calc(self.data, self.x, self.u)
        self.COST.calcDiff(self.data, self.x, self.u)

        self.cost_sum.calc(self.data_sum, self.x, self.u)
        self.cost_sum.calcDiff(self.data_sum, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.data.cost, self.data_sum.cost, 10, "Wrong cost value.")
        # Checking the Jacobians and Hessians of the cost
        self.assertTrue(np.allclose(self.data.Lx, self.data_sum.Lx, atol=1e-9), "Wrong Lx.")
        self.assertTrue(np.allclose(self.data.Lu, self.data_sum.Lu, atol=1e-9), "Wrong Lu.")
        self.assertTrue(np.allclose(self.data.Lxx, self.data_sum.Lxx, atol=1e-9), "Wrong Lxx.")
        self.assertTrue(np.allclose(self.data.Lxu, self.data_sum.Lxu, atol=1e-9), "Wrong Lxu.")
        self.assertTrue(np.allclose(self.data.Luu, self.data_sum.Luu, atol=1e-9), "Wrong Luu.")

    def test_removeCost(self):
        self.cost_sum.removeCost("myCost")
        self.assertEqual(len(self.cost_sum.costs), 0, "The number of cost items should be zero")


class StateCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(ROBOT_STATE, crocoddyl.ResidualModelState(ROBOT_STATE))
    COST_DER = StateCostModelDerived(ROBOT_STATE)


class StateCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(ROBOT_STATE, crocoddyl.ResidualModelState(ROBOT_STATE))


class ControlCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(ROBOT_STATE, crocoddyl.ResidualModelControl(ROBOT_STATE))
    COST_DER = ControlCostModelDerived(ROBOT_STATE)


class ControlCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(ROBOT_STATE, crocoddyl.ResidualModelControl(ROBOT_STATE))


class CoMPositionCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    cref = pinocchio.utils.rand(3)
    COST = crocoddyl.CostModelResidual(ROBOT_STATE, crocoddyl.ResidualModelCoMPosition(ROBOT_STATE, cref))
    COST_DER = CoMPositionCostModelDerived(ROBOT_STATE, cref=cref)


class CoMPositionCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    cref = pinocchio.utils.rand(3)
    COST = crocoddyl.CostModelResidual(ROBOT_STATE, crocoddyl.ResidualModelCoMPosition(ROBOT_STATE, cref))


class FramePlacementCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    Mref = pinocchio.SE3.Random()
    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE, crocoddyl.ResidualModelFramePlacement(ROBOT_STATE, ROBOT_MODEL.getFrameId('r_sole'), Mref))
    COST_DER = FramePlacementCostModelDerived(ROBOT_STATE, frame_id=ROBOT_MODEL.getFrameId('r_sole'), placement=Mref)


class FramePlacementCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFramePlacement(ROBOT_STATE, ROBOT_MODEL.getFrameId('r_sole'), pinocchio.SE3.Random()))


class FrameTranslationCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    xref = pinocchio.utils.rand(3)
    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE, crocoddyl.ResidualModelFrameTranslation(ROBOT_STATE, ROBOT_MODEL.getFrameId('r_sole'), xref))
    COST_DER = FrameTranslationCostModelDerived(ROBOT_STATE,
                                                frame_id=ROBOT_MODEL.getFrameId('r_sole'),
                                                translation=xref)


class FrameTranslationCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFrameTranslation(ROBOT_STATE, ROBOT_MODEL.getFrameId('r_sole'),
                                                pinocchio.utils.rand(3)))


class FrameRotationCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    Rref = pinocchio.SE3.Random().rotation
    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE, crocoddyl.ResidualModelFrameRotation(ROBOT_STATE, ROBOT_MODEL.getFrameId('r_sole'), Rref))
    COST_DER = FrameRotationCostModelDerived(ROBOT_STATE, frame_id=ROBOT_MODEL.getFrameId('r_sole'), rotation=Rref)


class FrameRotationCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFrameRotation(ROBOT_STATE, ROBOT_MODEL.getFrameId('r_sole'),
                                             pinocchio.SE3.Random().rotation))


class FrameVelocityCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    vref = pinocchio.Motion.Random()
    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFrameVelocity(ROBOT_STATE, ROBOT_MODEL.getFrameId('r_sole'), vref, pinocchio.LOCAL))
    COST_DER = FrameVelocityCostModelDerived(ROBOT_STATE, frame_id=ROBOT_MODEL.getFrameId('r_sole'), velocity=vref)


class FrameVelocityCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load('icub_reduced').model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFrameVelocity(ROBOT_STATE, ROBOT_MODEL.getFrameId('r_sole'), pinocchio.Motion.Random(),
                                             pinocchio.LOCAL))


if __name__ == '__main__':
    test_classes_to_run = [
        StateCostTest, StateCostSumTest, ControlCostTest, ControlCostSumTest, CoMPositionCostTest,
        CoMPositionCostSumTest, FramePlacementCostTest, FramePlacementCostSumTest, FrameTranslationCostTest,
        FrameTranslationCostSumTest, FrameRotationCostTest, FrameRotationCostSumTest, FrameVelocityCostTest,
        FrameVelocityCostSumTest
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
