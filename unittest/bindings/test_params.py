###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################

import unittest

import numpy as np

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


class ParamsOverride(crocoddyl.ParamsAbstract):
    def __init__(self, state, np_=3):
        super().__init__(state, np_)
        self.update_calls = 0
        self.create_calls = 0

    def update(self, data, p):
        self.update_calls += 1
        data.p = p

    def createData(self):
        self.create_calls += 1
        data = super().createData()
        data.active = False
        return data


class ActionParamsOverride(crocoddyl.ActionModelParamsAbstract):
    def __init__(self, state, np_=3):
        super().__init__(state, np_)
        self.update_calls = 0
        self.create_calls = 0
        self.sensitivity_calls = 0

    def update(self, data, p):
        self.update_calls += 1
        data.p = p

    def createData(self):
        self.create_calls += 1
        data = crocoddyl.ActionModelParamsDataAbstract(self)
        data.active = False
        return data

    def computeParamSensitivity(self, data, params, x, u):
        self.sensitivity_calls += 1
        params.dx_dp = np.full((self.state.ndx, self.np), x.sum() + u.sum())


class ParamsTest(unittest.TestCase):
    def setUp(self):
        self.state = crocoddyl.StateVector(4)

    def test_base_defaults_bounds_update_data_copy_and_errors(self):
        empty = crocoddyl.ParamsAbstract(self.state)
        self.assertEqual(empty.np, 0)
        self.assertEqual(empty.zero().shape, (0,))

        model = crocoddyl.ParamsAbstract(self.state, 3)
        self.assertEqual(model.state.nx, self.state.nx)
        self.assertEqual(model.np, 3)
        self.assertEqual(model.lb.shape, (3,))
        self.assertEqual(model.ub.shape, (3,))
        self.assertTrue(np.all(model.lb < 0.0))
        self.assertTrue(np.all(model.ub > 0.0))
        self.assertTrue(np.array_equal(model.zero(), np.zeros(3)))
        random = model.rand()
        self.assertEqual(random.shape, (3,))
        self.assertTrue(np.all(random >= 0.0))
        self.assertTrue(np.all(random <= 1.0))

        lb = np.array([-3.0, -2.0, -1.0])
        ub = np.array([1.0, 2.0, 3.0])
        model.lb = lb
        model.ub = ub
        self.assertTrue(np.array_equal(model.lb, lb))
        self.assertTrue(np.array_equal(model.ub, ub))
        with self.assertRaises(Exception):
            model.lb = np.zeros(4)
        with self.assertRaises(Exception):
            model.ub = np.zeros(4)

        data = model.createData()
        self.assertEqual((data.np, data.np_action, data.np_dynamics), (3, 3, 0))
        self.assertTrue(model.checkData(data))
        self.assertFalse(
            model.checkData(crocoddyl.ParamsDataAbstract(self.state, 4, 0))
        )
        data.p = np.ones(3)
        model.update(data, np.zeros(3))
        self.assertTrue(np.array_equal(data.p, np.ones(3)))

        self.assertIn("ParamsAbstract", repr(model))

    def test_base_python_override_and_fallback(self):
        model = ParamsOverride(self.state)
        data = model.createData()
        self.assertEqual(model.create_calls, 1)
        self.assertFalse(data.active)
        p = np.array([0.1, 0.2, 0.3])
        model.update(data, p)
        self.assertEqual(model.update_calls, 1)
        self.assertTrue(np.array_equal(data.p, p))

        crocoddyl.ParamsAbstract.update(model, data, np.zeros(3))
        self.assertTrue(np.array_equal(data.p, p))
        fallback_data = crocoddyl.ParamsAbstract.createData(model)
        self.assertTrue(fallback_data.active)
        self.assertEqual(
            (fallback_data.np, fallback_data.np_action, fallback_data.np_dynamics),
            (3, 3, 0),
        )

    def test_action_override_model_data_and_base_conversion(self):
        model = ActionParamsOverride(self.state)
        model.lb = np.full(3, -2.0)
        model.ub = np.full(3, 2.0)
        params = model.createData()
        self.assertEqual(model.create_calls, 1)
        self.assertIsInstance(params, crocoddyl.ActionModelParamsDataAbstract)
        self.assertFalse(params.active)
        fallback_params = crocoddyl.ActionModelParamsAbstract.createData(model)
        self.assertIsInstance(fallback_params, crocoddyl.ActionModelParamsDataAbstract)
        self.assertTrue(fallback_params.active)
        p = np.array([0.2, 0.4, 0.6])
        model.update(params, p)
        self.assertEqual(model.update_calls, 1)
        self.assertTrue(np.array_equal(params.p, p))

        action = crocoddyl.ActionModelLQR(4, 2)
        action_data = action.createData()
        x = np.array([0.1, 0.2, 0.3, 0.4])
        u = np.array([0.5, 0.6])
        model.computeParamSensitivity(action_data, params, x, u)
        self.assertEqual(model.sensitivity_calls, 1)
        self.assertTrue(np.allclose(params.dx_dp, x.sum() + u.sum()))

        action_data_payload = crocoddyl.ActionModelParamsDataAbstract(model)
        base_data_from_derived = crocoddyl.ParamsDataAbstract(model)
        for data in (action_data_payload, base_data_from_derived):
            self.assertEqual((data.np, data.np_action, data.np_dynamics), (3, 3, 0))
            data.p = p
            data.dx_dp = np.ones((self.state.ndx, 3))
            data.dtau_dp = np.empty((self.state.nv, 0))
            self.assertTrue(np.array_equal(data.p, p))
            self.assertTrue(np.array_equal(data.dx_dp, np.ones((self.state.ndx, 3))))

        model.update(action_data_payload, p)
        self.assertTrue(model.checkData(action_data_payload))
        collector = crocoddyl.DataCollectorParams(action_data_payload)
        self.assertIs(collector.params, action_data_payload)

        with self.assertRaises(TypeError):
            crocoddyl.ActionModelParamsDataAbstract(self.state, 3)
        with self.assertRaises(Exception):
            crocoddyl.ParamsDataAbstract(None)
        with self.assertRaises(Exception):
            crocoddyl.ActionModelParamsDataAbstract(None)

        default_model = crocoddyl.ActionModelParamsAbstract(self.state, 3)
        self.assertEqual(default_model.np, 3)
        self.assertEqual(default_model.state.nx, self.state.nx)
        default_data = default_model.createData()
        default_data.p = np.ones(3)
        default_model.update(default_data, np.zeros(3))
        self.assertTrue(np.array_equal(default_data.p, np.ones(3)))
        with self.assertRaises(RuntimeError):
            default_model.computeParamSensitivity(
                action_data, default_data, np.zeros(4), np.zeros(2)
            )

    def test_float32_models_overrides_and_final_data_constructor(self):
        state = crocoddyl_float32.StateVector(4)
        model = crocoddyl_float32.ParamsAbstract(state, 2)
        model.lb = np.array([-2.0, -1.0], dtype=np.float32)
        model.ub = np.array([1.0, 2.0], dtype=np.float32)
        data = model.createData()
        self.assertEqual(data.p.dtype, np.float32)
        self.assertEqual((data.np, data.np_action, data.np_dynamics), (2, 2, 0))

        class FloatActionParams(crocoddyl_float32.ActionModelParamsAbstract):
            def __init__(self, state_, np_):
                super().__init__(state_, np_)
                self.create_calls = 0

            def createData(self):
                self.create_calls += 1
                data_ = crocoddyl_float32.ActionModelParamsDataAbstract(self)
                data_.active = False
                return data_

            def computeParamSensitivity(self, data, params, x, u):
                params.dx_dp = np.full((self.state.ndx, self.np), 4.0, dtype=np.float32)

        action_model = FloatActionParams(state, 2)
        action_params = action_model.createData()
        self.assertEqual(action_model.create_calls, 1)
        self.assertIsInstance(
            action_params, crocoddyl_float32.ActionModelParamsDataAbstract
        )
        self.assertFalse(action_params.active)
        fallback_params = crocoddyl_float32.ActionModelParamsAbstract.createData(
            action_model
        )
        self.assertIsInstance(
            fallback_params, crocoddyl_float32.ActionModelParamsDataAbstract
        )
        self.assertTrue(fallback_params.active)
        converted = crocoddyl_float32.ParamsDataAbstract(action_model)
        p = np.array([0.2, 0.4], dtype=np.float32)
        action_model.update(action_params, p)
        self.assertTrue(action_model.checkData(action_params))
        collector = crocoddyl_float32.DataCollectorParams(action_params)
        self.assertIs(collector.params, action_params)
        action = crocoddyl_float32.ActionModelLQR(4, 2)
        action_model.computeParamSensitivity(
            action.createData(),
            action_params,
            np.zeros(4, dtype=np.float32),
            np.zeros(2, dtype=np.float32),
        )
        self.assertTrue(np.array_equal(action_params.dx_dp, np.full((4, 2), 4.0)))
        self.assertEqual(converted.p.dtype, np.float32)

        bare_model = crocoddyl_float32.ActionModelParamsAbstract(state, 2)
        with self.assertRaises(RuntimeError):
            bare_model.computeParamSensitivity(
                action.createData(),
                bare_model.createData(),
                np.zeros(4, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            )


if __name__ == "__main__":
    unittest.main()
