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
    def __init__(self, np_=3):
        super().__init__(crocoddyl.StateVector(4), 2, 2, np_, False, False, False)

    def calc(self, data, x, u=None):
        data.r[:] = np.array([1.0, 2.0])

    def calcDiff(self, data, x, u=None):
        data.Rx[:] = 0.0
        data.Ru[:] = 0.0
        if self.np:
            data.Rp[:] = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])


class ParameterConstraint(crocoddyl.ConstraintModelAbstract):
    def __init__(self, inequality=False, np_=3, state_dependent=False):
        state = crocoddyl.StateVector(4)
        residual = crocoddyl.ResidualModelAbstract(
            state, 2, 2, np_, state_dependent, state_dependent, False
        )
        super().__init__(
            state, residual, 2 if inequality else 0, 0 if inequality else 2
        )

    def calc(self, data, x, u=None):
        if self.ng:
            data.g[:] = np.array([1.0, 2.0])
        else:
            data.h[:] = np.array([1.0, 2.0])

    def calcDiff(self, data, x, u=None):
        parameter_jacobian = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]])
        if self.ng:
            data.Gp[:] = parameter_jacobian
        else:
            data.Hp[:] = parameter_jacobian


class ParameterAction(crocoddyl.ActionModelAbstract):
    def __init__(self, ng, nh, np_):
        super().__init__(crocoddyl.StateVector(4), 2, 0, ng, nh, ng, nh, np_)

    def calc(self, data, x, u=None):
        data.xnext[:] = x

    def calcDiff(self, data, x, u=None):
        pass


class ConstraintParametersTest(unittest.TestCase):
    def setUp(self):
        self.residual = ParameterResidual()
        self.state = self.residual.state
        self.x = self.state.rand()
        self.u = np.array([0.2, -0.3])
        self.shared = crocoddyl.DataCollectorAbstract()

    def test_constraint_data_propagation_and_assignment(self):
        equality = crocoddyl.ConstraintModelResidual(self.state, self.residual)
        equality_data = equality.createData(self.shared)
        self.assertEqual(equality.np, 3)
        self.assertEqual(equality_data.Gp.shape, (0, 3))
        self.assertEqual(equality_data.Hp.shape, (2, 3))
        self.assertTrue(np.allclose(equality_data.Gp, 0.0))
        self.assertTrue(np.allclose(equality_data.Hp, 0.0))

        equality_data.Gp = np.empty((0, 3))
        equality_data.Hp = np.arange(6.0).reshape(2, 3)
        self.assertTrue(np.array_equal(equality_data.Hp, np.arange(6).reshape(2, 3)))

        lower = -np.ones(2)
        upper = np.ones(2)
        inequality = crocoddyl.ConstraintModelResidual(
            self.state, self.residual, lower, upper
        )
        inequality_data = inequality.createData(self.shared)
        inequality_data.Gp = np.full((2, 3), 4.0)
        inequality_data.Hp = np.empty((0, 3))
        self.assertTrue(np.allclose(inequality_data.Gp, 4.0))

    def test_manager_aggregation_resize_setters_and_copy(self):
        state_only = crocoddyl.ConstraintModelManager(self.state)
        legacy = crocoddyl.ConstraintModelManager(self.state, 2)
        parameterized = crocoddyl.ConstraintModelManager(self.state, 2, 3)
        self.assertEqual(state_only.np, 0)
        self.assertEqual(legacy.np, 0)
        self.assertEqual(parameterized.np, 3)

        manager = crocoddyl.ConstraintModelManager(self.state, 2)
        parameter_equality = ParameterConstraint()
        parameter_inequality = ParameterConstraint(inequality=True)
        control = crocoddyl.ResidualModelControl(self.state, 2)
        control_equality = crocoddyl.ConstraintModelResidual(self.state, control)
        manager.addConstraint("a_parameter_equality", parameter_equality)
        manager.addConstraint("b_parameter_inequality", parameter_inequality)
        manager.addConstraint("c_control_equality", control_equality)
        manager.addConstraint("d_inactive", parameter_equality, False)
        self.assertEqual(manager.np, 3)
        self.assertIn("d_inactive", manager.inactive_set)

        data = manager.createData(self.shared)
        self.assertEqual(data.Gp.shape, (manager.ng, manager.np))
        self.assertEqual(data.Hp.shape, (manager.nh, manager.np))
        self.assertTrue(np.allclose(data.Gp, 0.0))
        self.assertTrue(np.allclose(data.Hp, 0.0))
        manager.calc(data, self.x, self.u)
        data.Gp = np.full((manager.ng, manager.np), 42.0)
        data.Hp = np.full((manager.nh, manager.np), 42.0)
        manager.calcDiff(data, self.x, self.u)
        self.assertTrue(
            np.allclose(data.Gp, data.constraints["b_parameter_inequality"].Gp)
        )
        self.assertTrue(
            np.allclose(data.Hp[:2], data.constraints["a_parameter_equality"].Hp)
        )
        self.assertTrue(np.allclose(data.Hp[2:], 0.0))

        data.Gp = np.ones((manager.ng, manager.np))
        data.Hp = np.full((manager.nh, manager.np), 2.0)
        with self.assertRaises(crocoddyl.Exception):
            data.Gp = np.zeros((manager.ng, manager.np + 1))
        with self.assertRaises(crocoddyl.Exception):
            data.Hp = np.zeros((manager.nh, manager.np + 1))
        data_copy = copy.copy(data)
        model_copy = copy.copy(manager)
        self.assertTrue(np.array_equal(data_copy.Gp, data.Gp))
        self.assertTrue(np.array_equal(data_copy.Hp, data.Hp))
        self.assertEqual(model_copy.np, manager.np)

        data.resize(manager, False)
        manager.calc(data, self.x)
        manager.calcDiff(data, self.x)
        self.assertEqual(data.Gp.shape, (manager.ng_T, manager.np))
        self.assertEqual(data.Hp.shape, (manager.nh_T, manager.np))

        mismatch = crocoddyl.ConstraintModelManager(self.state, 2, 3)
        with self.assertRaises(crocoddyl.Exception):
            mismatch.addConstraint("bad", ParameterConstraint(np_=4))
        mismatch.addConstraint("parameter_free", control_equality)
        self.assertEqual(mismatch.np, 3)

    def test_full_parameter_storage_sharing(self):
        manager = crocoddyl.ConstraintModelManager(self.state, 2, 3)
        manager.addConstraint("equality", ParameterConstraint())
        manager.addConstraint("inequality", ParameterConstraint(inequality=True))
        data = manager.createData(self.shared)

        action = ParameterAction(manager.ng, manager.nh, manager.np)
        action_data = action.createData()
        data.shareMemory(action_data)
        data.Gp = np.ones((manager.ng, manager.np))
        data.Hp = np.full((manager.nh, manager.np), 2.0)
        self.assertTrue(np.array_equal(action_data.Gp, data.Gp))
        self.assertTrue(np.array_equal(action_data.Hp, data.Hp))

        incompatible = ParameterAction(manager.ng, manager.nh, manager.np + 1)
        with self.assertRaises(crocoddyl.Exception):
            data.shareMemory(incompatible.createData())


if __name__ == "__main__":
    unittest.main()
