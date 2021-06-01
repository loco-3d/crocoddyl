import sys
import unittest
from random import randint

import example_robot_data
import numpy as np

import crocoddyl
import pinocchio
from crocoddyl.utils import (DifferentialFreeFwdDynamicsModelDerived, DifferentialLQRModelDerived, LQRModelDerived,
                             UnicycleModelDerived, IntegratedActionModelRK4Derived)


class ActionModelAbstractTestCase(unittest.TestCase):
    MODEL = None
    MODEL_DER = None

    def setUp(self):
        state = self.MODEL.state
        self.x = state.rand()
        self.u = np.random.rand(self.MODEL.nu)
        self.DATA = self.MODEL.createData()
        self.DATA_DER = self.MODEL_DER.createData()

    def test_calc(self):
        # Run calc for both action models
        self.MODEL.calc(self.DATA, self.x, self.u)
        self.MODEL_DER.calc(self.DATA_DER, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(self.DATA.cost, self.DATA_DER.cost, 10, "Wrong cost value.")
        self.assertTrue(np.allclose(self.DATA.r, self.DATA_DER.r, atol=1e-9), "Wrong cost residuals.")

        if isinstance(self.MODEL, crocoddyl.ActionModelAbstract):
            # Checking the dimension of the next state
            self.assertEqual(self.DATA.xnext.shape, self.DATA_DER.xnext.shape, "Wrong next state dimension.")
            # Checking the next state value
            self.assertTrue(np.allclose(self.DATA.xnext, self.DATA_DER.xnext, atol=1e-9), "Wrong next state.")
        elif isinstance(self.MODEL, crocoddyl.DifferentialActionModelAbstract):
            # Checking the dimension of the next state
            self.assertEqual(self.DATA.xout.shape, self.DATA_DER.xout.shape, "Wrong next state dimension.")
            # Checking the next state value
            self.assertTrue(np.allclose(self.DATA.xout, self.DATA_DER.xout, atol=1e-9), "Wrong next state.")

    def test_calcDiff(self):
        # Run calcDiff for both action models
        self.MODEL.calc(self.DATA, self.x, self.u)
        self.MODEL.calcDiff(self.DATA, self.x, self.u)

        self.MODEL_DER.calc(self.DATA_DER, self.x, self.u)
        self.MODEL_DER.calcDiff(self.DATA_DER, self.x, self.u)
        # Checking the next state value
        if isinstance(self.MODEL, crocoddyl.ActionModelAbstract):
            self.assertTrue(np.allclose(self.DATA.xnext, self.DATA_DER.xnext, atol=1e-9), "Wrong next state.")
        elif isinstance(self.MODEL, crocoddyl.DifferentialActionModelAbstract):
            self.assertTrue(np.allclose(self.DATA.xout, self.DATA_DER.xout, atol=1e-9), "Wrong next state.")
        # Checking the Jacobians of the dynamic
        self.assertTrue(np.allclose(self.DATA.Fx, self.DATA_DER.Fx, atol=1e-9), "Wrong Fx.")
        self.assertTrue(np.allclose(self.DATA.Fu, self.DATA_DER.Fu, atol=1e-9), "Wrong Fu.")
        # Checking the Jacobians and Hessians of the cost
        self.assertTrue(np.allclose(self.DATA.Lx, self.DATA_DER.Lx, atol=1e-9), "Wrong Lx.")
        self.assertTrue(np.allclose(self.DATA.Lu, self.DATA_DER.Lu, atol=1e-9), "Wrong Lu.")
        self.assertTrue(np.allclose(self.DATA.Lxx, self.DATA_DER.Lxx, atol=1e-9), "Wrong Lxx.")
        self.assertTrue(np.allclose(self.DATA.Lxu, self.DATA_DER.Lxu, atol=1e-9), "Wrong Lxu.")
        self.assertTrue(np.allclose(self.DATA.Luu, self.DATA_DER.Luu, atol=1e-9), "Wrong Luu.")


class UnicycleTest(ActionModelAbstractTestCase):
    MODEL = crocoddyl.ActionModelUnicycle()
    MODEL_DER = UnicycleModelDerived()


class LQRTest(ActionModelAbstractTestCase):
    NX = randint(2, 21)
    NU = randint(2, NX)
    MODEL = crocoddyl.ActionModelLQR(NX, NU)
    MODEL_DER = LQRModelDerived(NX, NU)


class DifferentialLQRTest(ActionModelAbstractTestCase):
    NX = randint(2, 21)
    NU = randint(2, NX)
    MODEL = crocoddyl.DifferentialActionModelLQR(NX, NU)
    MODEL_DER = DifferentialLQRModelDerived(NX, NU)


class TalosArmFreeFwdDynamicsTest(ActionModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load('talos_arm').model
    STATE = crocoddyl.StateMultibody(ROBOT_MODEL)
    ACTUATION = crocoddyl.ActuationModelFull(STATE)
    COST_SUM = crocoddyl.CostModelSum(STATE)
    COST_SUM.addCost(
        'gripperPose',
        crocoddyl.CostModelResidual(
            STATE,
            crocoddyl.ResidualModelFramePlacement(STATE, ROBOT_MODEL.getFrameId("gripper_left_joint"),
                                                  pinocchio.SE3.Random())), 1e-3)
    COST_SUM.addCost("xReg", crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelState(STATE)), 1e-7)
    COST_SUM.addCost("uReg", crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelControl(STATE)), 1e-7)
    MODEL = crocoddyl.DifferentialActionModelFreeFwdDynamics(STATE, ACTUATION, COST_SUM)
    MODEL_DER = DifferentialFreeFwdDynamicsModelDerived(STATE, ACTUATION, COST_SUM)


class TalosArmFreeFwdDynamicsWithArmatureTest(ActionModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load('talos_arm').model
    STATE = crocoddyl.StateMultibody(ROBOT_MODEL)
    ACTUATION = crocoddyl.ActuationModelFull(STATE)
    COST_SUM = crocoddyl.CostModelSum(STATE)
    COST_SUM.addCost(
        'gripperPose',
        crocoddyl.CostModelResidual(
            STATE,
            crocoddyl.ResidualModelFramePlacement(STATE, ROBOT_MODEL.getFrameId("gripper_left_joint"),
                                                  pinocchio.SE3.Random())), 1e-3)
    COST_SUM.addCost("xReg", crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelState(STATE)), 1e-7)
    COST_SUM.addCost("uReg", crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelControl(STATE)), 1e-7)
    MODEL = crocoddyl.DifferentialActionModelFreeFwdDynamics(STATE, ACTUATION, COST_SUM)
    MODEL_DER = DifferentialFreeFwdDynamicsModelDerived(STATE, ACTUATION, COST_SUM)
    MODEL.armature = 0.1 * np.ones(ROBOT_MODEL.nv)
    MODEL_DER.set_armature(0.1 * np.ones(ROBOT_MODEL.nv))


class AnymalFreeFwdDynamicsTest(ActionModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load('anymal').model
    STATE = crocoddyl.StateMultibody(ROBOT_MODEL)
    ACTUATION = crocoddyl.ActuationModelFloatingBase(STATE)
    COST_SUM = crocoddyl.CostModelSum(STATE, ACTUATION.nu)
    COST_SUM.addCost("xReg", crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelState(STATE, ACTUATION.nu)),
                     1e-7)
    COST_SUM.addCost("uReg", crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelControl(STATE, ACTUATION.nu)),
                     1e-7)
    MODEL = crocoddyl.DifferentialActionModelFreeFwdDynamics(STATE, ACTUATION, COST_SUM)
    MODEL_DER = DifferentialFreeFwdDynamicsModelDerived(STATE, ACTUATION, COST_SUM)


class TalosArmIntegratedRK4Test(ActionModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load('talos_arm').model
    STATE = crocoddyl.StateMultibody(ROBOT_MODEL)
    ACTUATION = crocoddyl.ActuationModelFull(STATE)
    COST_SUM = crocoddyl.CostModelSum(STATE)
    COST_SUM.addCost(
        'gripperPose',
        crocoddyl.CostModelResidual(
            STATE,
            crocoddyl.ResidualModelFramePlacement(STATE, ROBOT_MODEL.getFrameId("gripper_left_joint"),
                                                  pinocchio.SE3.Random())), 1e-3)
    COST_SUM.addCost("xReg", crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelState(STATE)), 1e-7)
    COST_SUM.addCost("uReg", crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelControl(STATE)), 1e-7)
    DIFFERENTIAL = crocoddyl.DifferentialActionModelFreeFwdDynamics(STATE, ACTUATION, COST_SUM)
    MODEL = crocoddyl.IntegratedActionModelRK4(DIFFERENTIAL, 1e-3)
    MODEL_DER = IntegratedActionModelRK4Derived(DIFFERENTIAL, 1e-3)


class AnymalIntegratedRK4Test(ActionModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load('anymal').model
    STATE = crocoddyl.StateMultibody(ROBOT_MODEL)
    ACTUATION = crocoddyl.ActuationModelFloatingBase(STATE)
    COST_SUM = crocoddyl.CostModelSum(STATE, ACTUATION.nu)
    COST_SUM.addCost("xReg", crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelState(STATE, ACTUATION.nu)),
                     1e-7)
    COST_SUM.addCost("uReg", crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelControl(STATE, ACTUATION.nu)),
                     1e-7)
    DIFFERENTIAL = crocoddyl.DifferentialActionModelFreeFwdDynamics(STATE, ACTUATION, COST_SUM)
    MODEL = crocoddyl.IntegratedActionModelRK4(DIFFERENTIAL, 1e-3)
    MODEL_DER = IntegratedActionModelRK4Derived(DIFFERENTIAL, 1e-3)


if __name__ == '__main__':
    test_classes_to_run = [
        UnicycleTest,
        LQRTest,
        DifferentialLQRTest,
        TalosArmFreeFwdDynamicsTest,
        TalosArmFreeFwdDynamicsWithArmatureTest,
        AnymalFreeFwdDynamicsTest,
        AnymalIntegratedRK4Test,
        TalosArmIntegratedRK4Test,
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
