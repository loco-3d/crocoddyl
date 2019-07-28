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
            self.cost = self.COST(self.ROBOT_MODEL, Mref)
            self.costDer = self.COST_DER(self.ROBOT_MODEL, Mref=Mref)
        elif self.COST is crocoddyl.CostModelState:
            self.cost = self.COST(self.ROBOT_MODEL, self.STATE)
            self.costDer = self.COST_DER(self.ROBOT_MODEL, self.STATE)
        self.DATA = self.cost.createData(self.ROBOT_DATA)
        self.DATA_DER = self.costDer.createData(self.ROBOT_DATA)

        nq = self.ROBOT_MODEL.nq
        pinocchio.forwardKinematics(self.ROBOT_MODEL, self.ROBOT_DATA, self.x[:nq], self.x[nq:])
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.ROBOT_DATA, self.x[:nq])
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.ROBOT_DATA)

    def test_dimensions(self):
        self.assertEqual(self.cost.nx, self.costDer.nx, "Wrong nx.")
        self.assertEqual(self.cost.ndx, self.costDer.ndx, "Wrong ndx.")
        self.assertEqual(self.cost.nu, self.costDer.nu, "Wrong nu.")
        self.assertEqual(self.cost.nq, self.costDer.nq, "Wrong nq.")
        self.assertEqual(self.cost.nv, self.costDer.nv, "Wrong nv.")
        self.assertEqual(self.cost.nr, self.costDer.nr, "Wrong nr.")

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


class StateCostTest(CostModelAbstractTestCase):
    CostModelAbstractTestCase.COST = crocoddyl.CostModelState
    CostModelAbstractTestCase.COST_DER = utils.StateCostDerived


class FramePlacementCostTest(CostModelAbstractTestCase):
    CostModelAbstractTestCase.COST = crocoddyl.CostModelFramePlacement
    CostModelAbstractTestCase.COST_DER = utils.FramePlacementCostDerived


if __name__ == '__main__':
    unittest.main()
