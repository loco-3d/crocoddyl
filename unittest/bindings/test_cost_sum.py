###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026, University of Edinburgh, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################

import copy
import unittest

import numpy as np

import crocoddyl


class ParameterResidual(crocoddyl.ResidualModelAbstract):
    def __init__(self, nu=2, np_=2):
        super().__init__(crocoddyl.StateVector(4), 2, nu, np_, True, True, True)

    def calc(self, data, x, u=None):
        data.r[:] = np.array([1.0, -2.0])

    def calcDiff(self, data, x, u=None):
        data.Rx[:] = 0.25
        data.Ru[:] = -0.5
        if self.np:
            data.Rp[:] = np.array([[1.0, 2.0], [-1.0, 3.0]])


class ParameterAction(crocoddyl.ActionModelAbstract):
    def __init__(self, np_=2):
        super().__init__(crocoddyl.StateVector(4), 2, 0, 0, 0, 0, 0, np_)

    def calc(self, data, x, u=None):
        data.xnext[:] = x

    def calcDiff(self, data, x, u=None):
        pass


class ParameterCost(crocoddyl.CostModelAbstract):
    def __init__(self, residual):
        super().__init__(residual.state, residual)

    def calc(self, data, x, u=None):
        data.cost = 2.5

    def calcDiff(self, data, x, u=None):
        data.Lx[:] = 0.25
        data.Lxx[:] = 0.5
        if u is not None:
            data.Lu[:] = 0.75
            data.Luu[:] = 1.0
            data.Lxu[:] = 1.25
            data.Lpu[:] = 2.25
        if self.np:
            data.Lp[:] = 1.5
            data.Lpp[:] = 1.75
            data.Lpx[:] = 2.0


class CostSumParametersTest(unittest.TestCase):
    def setUp(self):
        self.residual = ParameterResidual()
        self.cost = ParameterCost(self.residual)
        self.model = crocoddyl.CostModelSum(
            self.residual.state, self.residual.nu, self.residual.np
        )
        self.model.addCost("active", self.cost, 2.0)
        self.model.addCost("inactive", self.cost, 3.0, False)
        self.data = self.model.createData(crocoddyl.DataCollectorAbstract())
        self.x = self.residual.state.rand()
        self.u = np.array([0.2, -0.3])

    def test_dimensions_initialization_and_setters(self):
        self.assertEqual(self.model.np, 2)
        self.assertEqual(self.data.Lp.shape, (2,))
        self.assertEqual(self.data.Lpp.shape, (2, 2))
        self.assertEqual(self.data.Lpx.shape, (2, self.residual.state.ndx))
        self.assertEqual(self.data.Lpu.shape, (2, self.residual.nu))
        self.assertTrue(np.allclose(self.data.Lp, 0.0))
        self.assertTrue(np.allclose(self.data.Lpp, 0.0))
        self.assertTrue(np.allclose(self.data.Lpx, 0.0))
        self.assertTrue(np.allclose(self.data.Lpu, 0.0))

        self.data.Lp = np.ones(2)
        self.data.Lpp = np.full((2, 2), 2.0)
        self.data.Lpx = np.full((2, 4), 3.0)
        self.data.Lpu = np.full((2, 2), 4.0)
        self.assertTrue(np.allclose(self.data.Lp, 1.0))
        self.assertTrue(np.allclose(self.data.Lpp, 2.0))
        self.assertTrue(np.allclose(self.data.Lpx, 3.0))
        self.assertTrue(np.allclose(self.data.Lpu, 4.0))
        with self.assertRaises(Exception):
            self.data.Lp = np.zeros(3)
        with self.assertRaises(Exception):
            self.data.Lpp = np.zeros((3, 3))
        with self.assertRaises(Exception):
            self.data.Lpx = np.zeros((3, 4))
        with self.assertRaises(Exception):
            self.data.Lpu = np.zeros((2, 3))

        data_copy = copy.copy(self.data)
        self.assertTrue(np.array_equal(data_copy.Lp, self.data.Lp))
        self.assertTrue(np.array_equal(data_copy.Lpp, self.data.Lpp))
        self.assertTrue(np.array_equal(data_copy.Lpx, self.data.Lpx))
        self.assertTrue(np.array_equal(data_copy.Lpu, self.data.Lpu))

    def test_running_terminal_and_active_status(self):
        self.assertIsNone(self.model.calc(self.data, self.x, self.u))
        self.assertIsNone(self.model.calcDiff(self.data, self.x, self.u))
        active_data = self.data.costs["active"]
        self.assertTrue(np.allclose(self.data.Lp, 2.0 * active_data.Lp))
        self.assertTrue(np.allclose(self.data.Lpp, 2.0 * active_data.Lpp))
        self.assertTrue(np.allclose(self.data.Lpx, 2.0 * active_data.Lpx))
        self.assertTrue(np.allclose(self.data.Lpu, 2.0 * active_data.Lpu))

        self.model.changeCostStatus("inactive", True)
        self.data.Lu = np.full(2, 41.0)
        self.data.Luu = np.full((2, 2), 42.0)
        self.data.Lxu = np.full((4, 2), 43.0)
        self.data.Lpu = np.full((2, 2), 44.0)
        self.model.calc(self.data, self.x)
        self.model.calcDiff(self.data, self.x)
        self.assertTrue(np.allclose(self.data.Lp, 5.0 * active_data.Lp))
        self.assertTrue(np.allclose(self.data.Lpp, 5.0 * active_data.Lpp))
        self.assertTrue(np.allclose(self.data.Lpx, 5.0 * active_data.Lpx))
        self.assertTrue(np.allclose(self.data.Lu, 41.0))
        self.assertTrue(np.allclose(self.data.Luu, 42.0))
        self.assertTrue(np.allclose(self.data.Lxu, 43.0))
        self.assertTrue(np.allclose(self.data.Lpu, 44.0))

    def test_dimension_validation_and_single_control_shapes(self):
        invalid = crocoddyl.CostModelSum(self.residual.state, 2, 3)
        with self.assertRaises(Exception):
            invalid.addCost("invalid", self.cost, 1.0)
        with self.assertRaises(Exception):
            invalid.addCost(crocoddyl.CostItem("invalid", self.cost, 1.0))

        parameter_free_residual = ParameterResidual(np_=0)
        parameter_free_cost = ParameterCost(parameter_free_residual)
        invalid.addCost("parameter_free", parameter_free_cost, 1.0)

        residual = ParameterResidual(nu=1)
        cost = ParameterCost(residual)
        model = crocoddyl.CostModelSum(residual.state, 1, 2)
        model.addCost("cost", cost, 1.0)
        data = model.createData(crocoddyl.DataCollectorAbstract())
        self.assertEqual(data.Lxu.shape, (4,))
        self.assertEqual(data.Lpu.shape, (2,))

    def test_full_share_memory_and_validation(self):
        action = ParameterAction()
        action_data = action.createData()
        self.data.shareMemory(action_data)
        self.data.Lp = np.ones(2)
        self.data.Lpp = np.full((2, 2), 2.0)
        self.data.Lpx = np.full((2, 4), 3.0)
        self.data.Lpu = np.full((2, 2), 4.0)
        self.assertTrue(np.allclose(action_data.Lp, self.data.Lp))
        self.assertTrue(np.allclose(action_data.Lpp, self.data.Lpp))
        self.assertTrue(np.allclose(action_data.Lpx, self.data.Lpx))
        self.assertTrue(np.allclose(action_data.Lpu, self.data.Lpu))

        incompatible_data = ParameterAction(3).createData()
        with self.assertRaises(Exception):
            self.data.shareMemory(incompatible_data)


if __name__ == "__main__":
    unittest.main()
