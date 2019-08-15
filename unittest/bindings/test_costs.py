import crocoddyl
import utils
import pinocchio
import numpy as np
import unittest
import sys


class CostModelAbstractTestCase(unittest.TestCase):
    ROBOT_MODEL = None
    ROBOT_STATE = None
    COST = None
    COST_DER = None

    def setUp(self):
        self.robot_data = self.ROBOT_MODEL.createData()
        self.x = self.ROBOT_STATE.rand()
        self.u = pinocchio.utils.rand(self.ROBOT_MODEL.nv)

        self.data = self.COST.createData(self.robot_data)
        self.data_der = self.COST_DER.createData(self.robot_data)

        nq = self.ROBOT_MODEL.nq
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:])
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data, self.x[:nq])
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.jacobianCenterOfMass(self.ROBOT_MODEL, self.robot_data, self.x[:nq], False)

    def test_dimensions(self):
        self.assertEqual(self.COST.State.nx, self.COST_DER.State.nx, "Wrong nx.")
        self.assertEqual(self.COST.State.ndx, self.COST_DER.State.ndx, "Wrong ndx.")
        self.assertEqual(self.COST.nu, self.COST_DER.nu, "Wrong nu.")
        self.assertEqual(self.COST.State.nq, self.COST_DER.State.nq, "Wrong nq.")
        self.assertEqual(self.COST.State.nv, self.COST_DER.State.nv, "Wrong nv.")
        self.assertEqual(self.COST.activation.nr, self.COST_DER.activation.nr, "Wrong nr.")

    def test_calc(self):
        # Run calc for both action models
        self.COST.calc(self.data, self.x, self.u)
        self.COST_DER.calc(self.data_der, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.data.cost, self.data_der.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.data.costResiduals, self.data_der.costResiduals, atol=1e-9),
                        "Wrong cost residuals.")

    def test_calcDiff(self):
        # Run calc for both action models
        self.COST.calcDiff(self.data, self.x, self.u)
        self.COST_DER.calcDiff(self.data_der, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.data.cost, self.data_der.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.data.costResiduals, self.data_der.costResiduals, atol=1e-9),
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

        self.data = self.COST.createData(self.robot_data)
        self.data_sum = self.cost_sum.createData(self.robot_data)

        nq = self.ROBOT_MODEL.nq
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:])
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data, self.x[:nq])
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.jacobianCenterOfMass(self.ROBOT_MODEL, self.robot_data, self.x[:nq], False)

    def test_dimensions(self):
        self.assertEqual(self.COST.State.nx, self.cost_sum.State.nx, "Wrong nx.")
        self.assertEqual(self.COST.State.ndx, self.cost_sum.State.ndx, "Wrong ndx.")
        self.assertEqual(self.COST.nu, self.cost_sum.nu, "Wrong nu.")
        self.assertEqual(self.COST.State.nq, self.cost_sum.State.nq, "Wrong nq.")
        self.assertEqual(self.COST.State.nv, self.cost_sum.State.nv, "Wrong nv.")
        self.assertEqual(self.COST.activation.nr, self.cost_sum.activation.nr, "Wrong nr.")

    def test_calc(self):
        # Run calc for both action models
        self.COST.calc(self.data, self.x, self.u)
        self.cost_sum.calc(self.data_sum, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.data.cost, self.data_sum.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.data.costResiduals, self.data_sum.costResiduals, atol=1e-9),
                        "Wrong cost residuals.")

    def test_calcDiff(self):
        # Run calc for both action models
        self.COST.calcDiff(self.data, self.x, self.u)
        self.cost_sum.calcDiff(self.data_sum, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.data.cost, self.data_sum.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.data.costResiduals, self.data_sum.costResiduals, atol=1e-9),
                        "Wrong cost residuals.")
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
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelState(ROBOT_STATE)
    COST_DER = utils.StateCostDerived(ROBOT_STATE)


class StateCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelState(ROBOT_STATE)


class ControlCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelControl(ROBOT_STATE)
    COST_DER = utils.ControlCostDerived(ROBOT_STATE)


class ControlCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelControl(ROBOT_STATE)


class CoMPositionCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    cref = pinocchio.utils.rand(3)
    COST = crocoddyl.CostModelCoMPosition(ROBOT_STATE, cref)
    COST_DER = utils.CoMPositionCostDerived(ROBOT_STATE, cref=cref)


class CoMPositionCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    cref = pinocchio.utils.rand(3)
    COST = crocoddyl.CostModelCoMPosition(ROBOT_STATE, cref)


class FramePlacementCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    Mref = crocoddyl.FramePlacement(ROBOT_MODEL.getFrameId('rleg5_joint'), pinocchio.SE3.Random())
    COST = crocoddyl.CostModelFramePlacement(ROBOT_STATE, Mref)
    COST_DER = utils.FramePlacementCostDerived(ROBOT_STATE, Mref=Mref)


class FramePlacementCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    Mref = crocoddyl.FramePlacement(ROBOT_MODEL.getFrameId('rleg5_joint'), pinocchio.SE3.Random())
    COST = crocoddyl.CostModelFramePlacement(ROBOT_STATE, Mref)


class FrameTranslationCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    xref = crocoddyl.FrameTranslation(ROBOT_MODEL.getFrameId('rleg5_joint'), pinocchio.utils.rand(3))
    COST = crocoddyl.CostModelFrameTranslation(ROBOT_STATE, xref)
    COST_DER = utils.FrameTranslationCostDerived(ROBOT_STATE, xref=xref)


class FrameTranslationCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    xref = crocoddyl.FrameTranslation(ROBOT_MODEL.getFrameId('rleg5_joint'), pinocchio.utils.rand(3))
    COST = crocoddyl.CostModelFrameTranslation(ROBOT_STATE, xref)


if __name__ == '__main__':
    test_classes_to_run = [
        StateCostTest, StateCostSumTest, ControlCostTest, ControlCostSumTest, CoMPositionCostTest,
        CoMPositionCostSumTest, FramePlacementCostTest, FramePlacementCostSumTest, FrameTranslationCostTest,
        FrameTranslationCostSumTest
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
