import crocoddyl
import utils
import pinocchio
import numpy as np
import unittest


class CostModelAbstractTestCase(unittest.TestCase):
    COST = None
    COST_DER = None

    def setUp(self):
        self.ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
        self.ROBOT_DATA = self.ROBOT_MODEL.createData()
        self.STATE = crocoddyl.StateMultibody(self.ROBOT_MODEL)
        self.x = self.STATE.rand()
        self.u = np.matrix(np.random.rand(self.ROBOT_MODEL.nv)).T

        if self.COST is crocoddyl.CostModelFramePlacement:
            Mref = crocoddyl.FramePlacement(self.ROBOT_MODEL.getFrameId('rleg5_joint'), pinocchio.SE3.Random())
            self.cost = self.COST(self.STATE, Mref)
            self.costDer = self.COST_DER(self.STATE, Mref=Mref)
        elif self.COST is crocoddyl.CostModelFrameTranslation:
            xref = crocoddyl.FrameTranslation(self.ROBOT_MODEL.getFrameId('rleg5_joint'),
                                              np.matrix(np.random.rand(3)).T)
            self.cost = self.COST(self.STATE, xref)
            self.costDer = self.COST_DER(self.STATE, xref=xref)
        elif self.COST is crocoddyl.CostModelControl:
            self.cost = self.COST(self.STATE)
            self.costDer = self.COST_DER(self.STATE)
        elif self.COST is crocoddyl.CostModelState:
            self.cost = self.COST(self.STATE)
            self.costDer = self.COST_DER(self.STATE)
        self.DATA = self.cost.createData(self.ROBOT_DATA)
        self.DATA_DER = self.costDer.createData(self.ROBOT_DATA)

        nq = self.ROBOT_MODEL.nq
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.ROBOT_DATA, self.x[:nq], self.x[nq:])
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.ROBOT_DATA, self.x[:nq])
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.ROBOT_DATA)

    def test_dimensions(self):
        self.assertEqual(self.cost.State.nx, self.costDer.State.nx, "Wrong nx.")
        self.assertEqual(self.cost.State.ndx, self.costDer.State.ndx, "Wrong ndx.")
        self.assertEqual(self.cost.nu, self.costDer.nu, "Wrong nu.")
        self.assertEqual(self.cost.State.nq, self.costDer.State.nq, "Wrong nq.")
        self.assertEqual(self.cost.State.nv, self.costDer.State.nv, "Wrong nv.")
        self.assertEqual(self.cost.activation.nr, self.costDer.activation.nr, "Wrong nr.")

    def test_calc(self):
        # Run calc for both action models
        self.cost.calc(self.DATA, self.x, self.u)
        self.costDer.calc(self.DATA_DER, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.DATA.cost, self.DATA_DER.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.DATA.costResiduals, self.DATA_DER.costResiduals, atol=1e-9),
                        "Wrong cost residuals.")

    def test_calcDiff(self):
        # Run calc for both action models
        self.cost.calcDiff(self.DATA, self.x, self.u)
        self.costDer.calcDiff(self.DATA_DER, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.DATA.cost, self.DATA_DER.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.DATA.costResiduals, self.DATA_DER.costResiduals, atol=1e-9),
                        "Wrong cost residuals.")
        # Checking the Jacobians and Hessians of the cost
        self.assertTrue(np.allclose(self.DATA.Lx, self.DATA_DER.Lx, atol=1e-9), "Wrong Lx.")
        self.assertTrue(np.allclose(self.DATA.Lu, self.DATA_DER.Lu, atol=1e-9), "Wrong Lu.")
        self.assertTrue(np.allclose(self.DATA.Lxx, self.DATA_DER.Lxx, atol=1e-9), "Wrong Lxx.")
        self.assertTrue(np.allclose(self.DATA.Lxu, self.DATA_DER.Lxu, atol=1e-9), "Wrong Lxu.")
        self.assertTrue(np.allclose(self.DATA.Luu, self.DATA_DER.Luu, atol=1e-9), "Wrong Luu.")


class CostModelSumTestCase(unittest.TestCase):
    COST = None

    def setUp(self):
        self.ROBOT_MODEL = pinocchio.buildSampleModelHumanoidRandom()
        self.ROBOT_DATA = self.ROBOT_MODEL.createData()
        self.STATE = crocoddyl.StateMultibody(self.ROBOT_MODEL)
        self.x = self.STATE.rand()
        self.u = np.matrix(np.random.rand(self.ROBOT_MODEL.nv)).T

        self.COST_SUM = crocoddyl.CostModelSum(self.STATE)

        if self.COST is crocoddyl.CostModelFramePlacement:
            Mref = crocoddyl.FramePlacement(self.ROBOT_MODEL.getFrameId('rleg5_joint'), pinocchio.SE3.Random())
            self.cost = self.COST(self.STATE, Mref)
        elif self.COST is crocoddyl.CostModelFrameTranslation:
            xref = crocoddyl.FrameTranslation(self.ROBOT_MODEL.getFrameId('rleg5_joint'),
                                              np.matrix(np.random.rand(3)).T)
            self.cost = self.COST(self.STATE, xref)
        elif self.COST is crocoddyl.CostModelControl:
            self.cost = self.COST(self.STATE)
        elif self.COST is crocoddyl.CostModelState:
            self.cost = self.COST(self.STATE)
        self.COST_SUM.addCost('myCost', self.cost, 1.)

        self.DATA = self.cost.createData(self.ROBOT_DATA)
        self.DATA_SUM = self.COST_SUM.createData(self.ROBOT_DATA)

        nq = self.ROBOT_MODEL.nq
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.ROBOT_DATA, self.x[:nq], self.x[nq:])
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.ROBOT_DATA, self.x[:nq])
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.ROBOT_DATA)

    def test_dimensions(self):
        self.assertEqual(self.cost.State.nx, self.COST_SUM.State.nx, "Wrong nx.")
        self.assertEqual(self.cost.State.ndx, self.COST_SUM.State.ndx, "Wrong ndx.")
        self.assertEqual(self.cost.nu, self.COST_SUM.nu, "Wrong nu.")
        self.assertEqual(self.cost.State.nq, self.COST_SUM.State.nq, "Wrong nq.")
        self.assertEqual(self.cost.State.nv, self.COST_SUM.State.nv, "Wrong nv.")
        self.assertEqual(self.cost.activation.nr, self.COST_SUM.activation.nr, "Wrong nr.")

    def test_calc(self):
        # Run calc for both action models
        self.cost.calc(self.DATA, self.x, self.u)
        self.COST_SUM.calc(self.DATA_SUM, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.DATA.cost, self.DATA_SUM.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.DATA.costResiduals, self.DATA_SUM.costResiduals, atol=1e-9),
                        "Wrong cost residuals.")

    def test_calcDiff(self):
        # Run calc for both action models
        self.cost.calcDiff(self.DATA, self.x, self.u)
        self.COST_SUM.calcDiff(self.DATA_SUM, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.DATA.cost, self.DATA_SUM.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.DATA.costResiduals, self.DATA_SUM.costResiduals, atol=1e-9),
                        "Wrong cost residuals.")
        # Checking the Jacobians and Hessians of the cost
        self.assertTrue(np.allclose(self.DATA.Lx, self.DATA_SUM.Lx, atol=1e-9), "Wrong Lx.")
        self.assertTrue(np.allclose(self.DATA.Lu, self.DATA_SUM.Lu, atol=1e-9), "Wrong Lu.")
        self.assertTrue(np.allclose(self.DATA.Lxx, self.DATA_SUM.Lxx, atol=1e-9), "Wrong Lxx.")
        self.assertTrue(np.allclose(self.DATA.Lxu, self.DATA_SUM.Lxu, atol=1e-9), "Wrong Lxu.")
        self.assertTrue(np.allclose(self.DATA.Luu, self.DATA_SUM.Luu, atol=1e-9), "Wrong Luu.")

    def test_removeCost(self):
        self.COST_SUM.removeCost("myCost")
        self.assertEqual(len(self.COST_SUM.costs), 0, "The number of cost items should be zero")


class StateCostTest(CostModelAbstractTestCase):
    COST = crocoddyl.CostModelState
    COST_DER = utils.StateCostDerived


class StateCostSumTest(CostModelSumTestCase):
    COST = crocoddyl.CostModelState


class ControlCostTest(CostModelAbstractTestCase):
    COST = crocoddyl.CostModelControl
    COST_DER = utils.ControlCostDerived


class ControlCostSumTest(CostModelSumTestCase):
    COST = crocoddyl.CostModelControl


class FramePlacementCostTest(CostModelAbstractTestCase):
    COST = crocoddyl.CostModelFramePlacement
    COST_DER = utils.FramePlacementCostDerived


class FramePlacementCostSumTest(CostModelSumTestCase):
    COST = crocoddyl.CostModelFramePlacement


class FrameTranslationCostTest(CostModelAbstractTestCase):
    COST = crocoddyl.CostModelFrameTranslation
    COST_DER = utils.FrameTranslationCostDerived


class FrameTranslationCostSumTest(CostModelSumTestCase):
    COST = crocoddyl.CostModelFrameTranslation


if __name__ == '__main__':
    test_classes_to_run = [
        StateCostTest, StateCostSumTest, ControlCostTest, ControlCostSumTest, FramePlacementCostTest,
        FramePlacementCostSumTest, FrameTranslationCostTest, FrameTranslationCostSumTest
    ]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
