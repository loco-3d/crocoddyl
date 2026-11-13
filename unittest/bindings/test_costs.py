import copy
import sys
import unittest

import example_robot_data
import numpy as np
import pinocchio
from factory import (
    CoMPositionCostModelDerived,
    ControlCostModelDerived,
    FramePlacementCostModelDerived,
    FrameRotationCostModelDerived,
    FrameTranslationCostModelDerived,
    FrameVelocityCostModelDerived,
    StateCostModelDerived,
)

import crocoddyl


class ParameterResidualDerived(crocoddyl.ResidualModelAbstract):
    def __init__(self, np_=2):
        super().__init__(crocoddyl.StateVector(4), 2, 2, np_, False, False, False)
        self.calc_calls = 0
        self.calc_diff_calls = 0
        self.cost_diff_calls = 0

    def calc(self, data, x, u=None):
        self.calc_calls += 1
        data.r[:] = np.array([1.0, -2.0])

    def calcDiff(self, data, x, u=None):
        self.calc_diff_calls += 1
        data.Rx[:] = 0.0
        data.Ru[:] = 0.0
        if self.np:
            data.Rp[:] = np.array([[1.0, 2.0], [-1.0, 3.0]])

    def calcCostDiff(self, cdata, rdata, adata, update_u=True):
        self.cost_diff_calls += 1
        return super().calcCostDiff(cdata, rdata, adata, update_u)


class ParameterResidualCostBindingsTest(unittest.TestCase):
    def test_constructor_data_copy_and_properties(self):
        residual = ParameterResidualDerived()
        collector = crocoddyl.DataCollectorAbstract()
        rdata = residual.createData(collector)

        self.assertEqual(residual.np, 2)
        self.assertEqual(residual.nu, 2)
        self.assertEqual(rdata.Rp.shape, (2, 2))
        self.assertEqual(rdata.Arr_Rp.shape, (2, 2))
        self.assertTrue(np.allclose(rdata.Rp, 0.0))
        self.assertTrue(np.allclose(rdata.Arr_Rp, 0.0))
        rdata.Rp = np.arange(4.0).reshape(2, 2)
        rdata.Arr_Rp[:] = 3.0
        rdata_copy = copy.copy(rdata)
        self.assertTrue(np.array_equal(rdata_copy.Rp, rdata.Rp))
        self.assertTrue(np.array_equal(rdata_copy.Arr_Rp, rdata.Arr_Rp))

        cost = crocoddyl.CostModelResidual(residual.state, residual)
        cdata = cost.createData(collector)
        self.assertEqual(cost.np, residual.np)
        self.assertEqual(cdata.Lp.shape, (2,))
        self.assertEqual(cdata.Lpp.shape, (2, 2))
        self.assertEqual(cdata.Lpx.shape, (2, residual.state.ndx))
        self.assertEqual(cdata.Lpu.shape, (2, residual.nu))
        cdata.Lp = np.array([1.0, 2.0])
        cdata.Lpp = np.full((2, 2), 3.0)
        cdata.Lpx = np.full((2, residual.state.ndx), 4.0)
        cdata.Lpu = np.full((2, residual.nu), 5.0)
        cdata_copy = copy.copy(cdata)
        self.assertTrue(np.array_equal(cdata_copy.Lp, cdata.Lp))
        self.assertTrue(np.array_equal(cdata_copy.Lpp, cdata.Lpp))
        self.assertTrue(np.array_equal(cdata_copy.Lpx, cdata.Lpx))
        self.assertTrue(np.array_equal(cdata_copy.Lpu, cdata.Lpu))

    def test_legacy_constructor_positions(self):
        state = crocoddyl.StateVector(4)
        residual = crocoddyl.ResidualModelAbstract(state, 2, 2, False, True, False)
        self.assertEqual(residual.np, 0)
        self.assertFalse(residual.q_dependent)
        self.assertTrue(residual.v_dependent)
        self.assertFalse(residual.u_dependent)

        default_nu_residual = crocoddyl.ResidualModelAbstract(
            state, 2, False, True, False
        )
        self.assertEqual(default_nu_residual.nu, state.nv)
        self.assertEqual(default_nu_residual.np, 0)
        self.assertFalse(default_nu_residual.q_dependent)
        self.assertTrue(default_nu_residual.v_dependent)
        self.assertFalse(default_nu_residual.u_dependent)

        parameter_residual = crocoddyl.ResidualModelAbstract(state, 2, None, 3)
        self.assertEqual(parameter_residual.nu, state.nv)
        self.assertEqual(parameter_residual.np, 3)

        parameter_residual = crocoddyl.ResidualModelAbstract(state, 2, np=3)
        self.assertEqual(parameter_residual.nu, state.nv)
        self.assertEqual(parameter_residual.np, 3)

    def test_parameter_only_running_terminal_and_wrapper_dispatch(self):
        residual = ParameterResidualDerived()
        cost = crocoddyl.CostModelResidual(residual.state, residual)
        data = cost.createData(crocoddyl.DataCollectorAbstract())
        x = residual.state.rand()
        u = np.array([0.2, -0.3])

        cost.calc(data, x, u)
        cost.calcDiff(data, x, u)
        expected_lp = data.residual.Rp.T @ data.residual.r
        expected_lpp = data.residual.Rp.T @ data.residual.Rp
        self.assertAlmostEqual(data.cost, 2.5)
        self.assertTrue(np.allclose(data.Lp, expected_lp))
        self.assertTrue(np.allclose(data.Lpp, expected_lpp))
        self.assertTrue(np.allclose(data.Lpx, 0.0))
        self.assertTrue(np.allclose(data.Lpu, 0.0))

        cost.calc(data, x)
        cost.calcDiff(data, x)
        self.assertEqual(residual.calc_calls, 2)
        self.assertEqual(residual.calc_diff_calls, 2)
        self.assertEqual(residual.cost_diff_calls, 2)
        self.assertTrue(np.allclose(data.Lp, expected_lp))
        self.assertTrue(np.allclose(data.Lpp, expected_lpp))


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
        pinocchio.forwardKinematics(
            self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:]
        )
        pinocchio.computeForwardKinematicsDerivatives(
            self.ROBOT_MODEL,
            self.robot_data,
            self.x[:nq],
            self.x[nq:],
            pinocchio.utils.zero(nv),
        )
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data, self.x[:nq])
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.jacobianCenterOfMass(
            self.ROBOT_MODEL, self.robot_data, self.x[:nq], False
        )

    def test_dimensions(self):
        self.assertEqual(self.COST.state.nx, self.COST_DER.state.nx, "Wrong nx.")
        self.assertEqual(self.COST.state.ndx, self.COST_DER.state.ndx, "Wrong ndx.")
        self.assertEqual(self.COST.nu, self.COST_DER.nu, "Wrong nu.")
        self.assertEqual(self.COST.state.nq, self.COST_DER.state.nq, "Wrong nq.")
        self.assertEqual(self.COST.state.nv, self.COST_DER.state.nv, "Wrong nv.")
        self.assertEqual(
            self.COST.activation.nr, self.COST_DER.activation.nr, "Wrong nr."
        )

    def test_calc(self):
        # Run calc for both action models
        self.COST.calc(self.data, self.x, self.u)
        self.COST_DER.calc(self.data_der, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(
            self.data.cost, self.data_der.cost, 10, "Wrong cost value."
        )
        self.assertTrue(
            np.allclose(self.data.residual.r, self.data_der.residual.r, atol=1e-9),
            "Wrong cost residuals.",
        )

    def test_calc_x(self):
        # Run calc for both action models
        self.COST.calc(self.data, self.x)
        self.COST_DER.calc(self.data_der, self.x)
        # Checking the cost value and its residual
        self.assertAlmostEqual(
            self.data.cost, self.data_der.cost, 10, "Wrong cost value."
        )
        self.assertTrue(
            np.allclose(self.data.residual.r, self.data_der.residual.r, atol=1e-9),
            "Wrong cost residuals.",
        )

    def test_calcDiff(self):
        # Run calc for both action models
        self.COST.calc(self.data, self.x, self.u)
        self.COST.calcDiff(self.data, self.x, self.u)

        self.COST_DER.calc(self.data_der, self.x, self.u)
        self.COST_DER.calcDiff(self.data_der, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(
            self.data.cost, self.data_der.cost, 10, "Wrong cost value."
        )
        self.assertTrue(
            np.allclose(self.data.residual.r, self.data_der.residual.r, atol=1e-9),
            "Wrong cost residuals.",
        )
        # Checking the Jacobians and Hessians of the cost
        self.assertTrue(
            np.allclose(self.data.Lx, self.data_der.Lx, atol=1e-9), "Wrong Lx."
        )
        self.assertTrue(
            np.allclose(self.data.Lu, self.data_der.Lu, atol=1e-9), "Wrong Lu."
        )
        self.assertTrue(
            np.allclose(self.data.Lxx, self.data_der.Lxx, atol=1e-9), "Wrong Lxx."
        )
        self.assertTrue(
            np.allclose(self.data.Lxu, self.data_der.Lxu, atol=1e-9), "Wrong Lxu."
        )
        self.assertTrue(
            np.allclose(self.data.Luu, self.data_der.Luu, atol=1e-9), "Wrong Luu."
        )

    def test_calcDiff_x(self):
        # Run calc for both action models
        self.COST.calc(self.data, self.x)
        self.COST.calcDiff(self.data, self.x)

        self.COST_DER.calc(self.data_der, self.x)
        self.COST_DER.calcDiff(self.data_der, self.x)
        # Checking the cost value and its residual
        self.assertAlmostEqual(
            self.data.cost, self.data_der.cost, 10, "Wrong cost value."
        )
        self.assertTrue(
            np.allclose(self.data.residual.r, self.data_der.residual.r, atol=1e-9),
            "Wrong cost residuals.",
        )
        # Checking the Jacobians and Hessians of the cost
        self.assertTrue(
            np.allclose(self.data.Lx, self.data_der.Lx, atol=1e-9), "Wrong Lx."
        )
        self.assertTrue(
            np.allclose(self.data.Lxx, self.data_der.Lxx, atol=1e-9), "Wrong Lxx."
        )


class CostModelSumTestCase(unittest.TestCase):
    ROBOT_MODEL = None
    ROBOT_STATE = None
    COST = None

    def setUp(self):
        self.robot_data = self.ROBOT_MODEL.createData()
        self.x = self.ROBOT_STATE.rand()
        self.u = pinocchio.utils.rand(self.ROBOT_MODEL.nv)

        self.cost_sum = crocoddyl.CostModelSum(self.ROBOT_STATE)
        self.cost_sum.addCost("myCost", self.COST, 1.0)

        self.multibody_data = crocoddyl.DataCollectorMultibody(self.robot_data)
        self.data = self.COST.createData(self.multibody_data)
        self.data_sum = self.cost_sum.createData(self.multibody_data)

        nq, nv = self.ROBOT_MODEL.nq, self.ROBOT_MODEL.nv
        pinocchio.forwardKinematics(
            self.ROBOT_MODEL, self.robot_data, self.x[:nq], self.x[nq:]
        )
        pinocchio.computeForwardKinematicsDerivatives(
            self.ROBOT_MODEL,
            self.robot_data,
            self.x[:nq],
            self.x[nq:],
            pinocchio.utils.zero(nv),
        )
        pinocchio.computeJointJacobians(self.ROBOT_MODEL, self.robot_data, self.x[:nq])
        pinocchio.updateFramePlacements(self.ROBOT_MODEL, self.robot_data)
        pinocchio.jacobianCenterOfMass(
            self.ROBOT_MODEL, self.robot_data, self.x[:nq], False
        )

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
        self.assertAlmostEqual(
            self.data.cost, self.data_sum.cost, 10, "Wrong cost value."
        )

    def test_calcDiff(self):
        # Run calc for both action models
        self.COST.calc(self.data, self.x, self.u)
        self.COST.calcDiff(self.data, self.x, self.u)

        self.cost_sum.calc(self.data_sum, self.x, self.u)
        self.cost_sum.calcDiff(self.data_sum, self.x, self.u)
        # Checking the cost value and its residual
        self.assertAlmostEqual(
            self.data.cost, self.data_sum.cost, 10, "Wrong cost value."
        )
        # Checking the Jacobians and Hessians of the cost
        self.assertTrue(
            np.allclose(self.data.Lx, self.data_sum.Lx, atol=1e-9), "Wrong Lx."
        )
        self.assertTrue(
            np.allclose(self.data.Lu, self.data_sum.Lu, atol=1e-9), "Wrong Lu."
        )
        self.assertTrue(
            np.allclose(self.data.Lxx, self.data_sum.Lxx, atol=1e-9), "Wrong Lxx."
        )
        self.assertTrue(
            np.allclose(self.data.Lxu, self.data_sum.Lxu, atol=1e-9), "Wrong Lxu."
        )
        self.assertTrue(
            np.allclose(self.data.Luu, self.data_sum.Luu, atol=1e-9), "Wrong Luu."
        )

    def test_removeCost(self):
        self.cost_sum.removeCost("myCost")
        self.assertEqual(
            len(self.cost_sum.costs), 0, "The number of cost items should be zero"
        )


class StateCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE, crocoddyl.ResidualModelState(ROBOT_STATE)
    )
    COST_DER = StateCostModelDerived(ROBOT_STATE)


class StateCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE, crocoddyl.ResidualModelState(ROBOT_STATE)
    )


class ControlCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE, crocoddyl.ResidualModelControl(ROBOT_STATE)
    )
    COST_DER = ControlCostModelDerived(ROBOT_STATE)


class ControlCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE, crocoddyl.ResidualModelControl(ROBOT_STATE)
    )


class CoMPositionCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    cref = pinocchio.utils.rand(3)
    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE, crocoddyl.ResidualModelCoMPosition(ROBOT_STATE, cref)
    )
    COST_DER = CoMPositionCostModelDerived(ROBOT_STATE, cref=cref)


class CoMPositionCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    cref = pinocchio.utils.rand(3)
    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE, crocoddyl.ResidualModelCoMPosition(ROBOT_STATE, cref)
    )


class FramePlacementCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    Mref = pinocchio.SE3.Random()
    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFramePlacement(
            ROBOT_STATE, ROBOT_MODEL.getFrameId("r_sole"), Mref
        ),
    )
    COST_DER = FramePlacementCostModelDerived(
        ROBOT_STATE, frame_id=ROBOT_MODEL.getFrameId("r_sole"), placement=Mref
    )


class FramePlacementCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFramePlacement(
            ROBOT_STATE, ROBOT_MODEL.getFrameId("r_sole"), pinocchio.SE3.Random()
        ),
    )


class FrameTranslationCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    xref = pinocchio.utils.rand(3)
    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFrameTranslation(
            ROBOT_STATE, ROBOT_MODEL.getFrameId("r_sole"), xref
        ),
    )
    COST_DER = FrameTranslationCostModelDerived(
        ROBOT_STATE, frame_id=ROBOT_MODEL.getFrameId("r_sole"), translation=xref
    )


class FrameTranslationCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFrameTranslation(
            ROBOT_STATE, ROBOT_MODEL.getFrameId("r_sole"), pinocchio.utils.rand(3)
        ),
    )


class FrameRotationCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    Rref = pinocchio.SE3.Random().rotation
    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFrameRotation(
            ROBOT_STATE, ROBOT_MODEL.getFrameId("r_sole"), Rref
        ),
    )
    COST_DER = FrameRotationCostModelDerived(
        ROBOT_STATE, frame_id=ROBOT_MODEL.getFrameId("r_sole"), rotation=Rref
    )


class FrameRotationCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFrameRotation(
            ROBOT_STATE,
            ROBOT_MODEL.getFrameId("r_sole"),
            pinocchio.SE3.Random().rotation,
        ),
    )


class FrameVelocityCostTest(CostModelAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    vref = pinocchio.Motion.Random()
    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFrameVelocity(
            ROBOT_STATE, ROBOT_MODEL.getFrameId("r_sole"), vref, pinocchio.LOCAL
        ),
    )
    COST_DER = FrameVelocityCostModelDerived(
        ROBOT_STATE, frame_id=ROBOT_MODEL.getFrameId("r_sole"), velocity=vref
    )


class FrameVelocityCostSumTest(CostModelSumTestCase):
    ROBOT_MODEL = example_robot_data.load("icub_reduced").model
    ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

    COST = crocoddyl.CostModelResidual(
        ROBOT_STATE,
        crocoddyl.ResidualModelFrameVelocity(
            ROBOT_STATE,
            ROBOT_MODEL.getFrameId("r_sole"),
            pinocchio.Motion.Random(),
            pinocchio.LOCAL,
        ),
    )


if __name__ == "__main__":
    # test to be run
    test_classes_to_run = [
        StateCostTest,
        StateCostSumTest,
        ControlCostTest,
        ControlCostSumTest,
        CoMPositionCostTest,
        CoMPositionCostSumTest,
        FramePlacementCostTest,
        FramePlacementCostSumTest,
        FrameTranslationCostTest,
        FrameTranslationCostSumTest,
        FrameRotationCostTest,
        FrameRotationCostSumTest,
        FrameVelocityCostTest,
        FrameVelocityCostSumTest,
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
