###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################

import copy
import unittest

import numpy as np

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


def make_action_params(module, dtype, state, np_, scale):
    class ActionParams(module.ActionModelParamsAbstract):
        def __init__(self):
            super().__init__(state, np_)
            self.scale = dtype(scale)
            self.update_calls = 0
            self.sensitivity_calls = 0

        def update(self, data, p):
            self.update_calls += 1
            data.p = np.asarray(p, dtype=dtype)

        def computeParamSensitivity(self, data, params, x, u):
            del data, params, x, u
            self.sensitivity_calls += 1
            columns = self.scale + np.arange(self.np, dtype=dtype)
            return np.tile(columns, (self.state.ndx, 1))

    return ActionParams()


def make_dynamics_params(module, dtype, state, np_, scale):
    class DynamicsParams(module.DynamicsParamsAbstract):
        def __init__(self):
            super().__init__(state, np_)
            self.scale = dtype(scale)
            self.update_calls = 0
            self.regressor_calls = 0

        def update(self, data, p):
            self.update_calls += 1
            data.p = np.asarray(p, dtype=dtype)

        def computeJointTorqueRegressor(self, data, params, x, u):
            del data, params, x, u
            self.regressor_calls += 1
            columns = self.scale + np.arange(self.np, dtype=dtype)
            return np.tile(columns, (self.state.nv, 1))

    return DynamicsParams()


class ParameterManagerTest(unittest.TestCase):
    def check_manager(self, module, dtype):
        state = module.StateVector(4)
        manager = module.ParameterManager(state)
        self.assertEqual(
            (manager.np, manager.np_action, manager.np_dynamics), (0, 0, 0)
        )
        self.assertEqual(list(manager.action_params.todict().keys()), [])
        self.assertEqual(list(manager.dynamics_params.todict().keys()), [])
        self.assertEqual(set(manager.active_set), set())
        self.assertEqual(set(manager.inactive_set), set())

        action_zeta = make_action_params(module, dtype, state, 1, 30)
        dynamics_zulu = make_dynamics_params(module, dtype, state, 2, 50)
        action_alpha = make_action_params(module, dtype, state, 2, 10)
        dynamics_idle = make_dynamics_params(module, dtype, state, 1, 70)
        dynamics_beta = make_dynamics_params(module, dtype, state, 1, 40)
        action_middle = make_action_params(module, dtype, state, 1, 20)
        manager.addParam("zeta", action_zeta)
        manager.addParam("zulu", dynamics_zulu)
        manager.addParam("alpha", action_alpha)
        manager.addParam("idle", dynamics_idle, False)
        manager.addParam("beta", dynamics_beta)
        manager.addParam("middle", action_middle, False)

        self.assertEqual(
            (manager.np, manager.np_action, manager.np_dynamics), (6, 3, 3)
        )
        self.assertEqual(
            list(manager.action_params.todict().keys()), ["alpha", "middle", "zeta"]
        )
        self.assertEqual(
            list(manager.dynamics_params.todict().keys()), ["beta", "idle", "zulu"]
        )
        self.assertEqual(set(manager.active_set), {"alpha", "beta", "zeta", "zulu"})
        self.assertEqual(set(manager.inactive_set), {"idle", "middle"})
        self.assertIs(manager.action_params["alpha"].param, action_alpha)
        self.assertIs(manager.dynamics_params["zulu"].param, dynamics_zulu)
        self.assertTrue(manager.action_params["alpha"].active)
        self.assertFalse(manager.action_params["middle"].active)
        alpha_item = manager.action_params["alpha"]
        replacements = (
            ("name", "replacement"),
            ("param", action_zeta),
            ("active", False),
        )
        for attribute, replacement in replacements:
            with self.assertRaises(AttributeError):
                setattr(alpha_item, attribute, replacement)
        self.assertEqual(alpha_item.name, "alpha")
        self.assertIs(alpha_item.param, action_alpha)
        self.assertTrue(alpha_item.active)
        replacement_lb = np.full(2, -3, dtype=dtype)
        alpha_item.param.lb = replacement_lb
        self.assertTrue(np.array_equal(action_alpha.lb, replacement_lb))
        manager.changeParamStatus("alpha", False)
        self.assertFalse(alpha_item.active)
        self.assertEqual((manager.np, manager.np_action), (4, 1))
        self.assertNotIn("alpha", set(manager.active_set))
        self.assertIn("alpha", set(manager.inactive_set))
        manager.changeParamStatus("alpha", True)
        self.assertTrue(alpha_item.active)
        self.assertEqual((manager.np, manager.np_action), (6, 3))
        self.assertIn("alpha", set(manager.active_set))
        self.assertNotIn("alpha", set(manager.inactive_set))

        data = manager.createData()
        direct_data = module.ParameterDataManager(manager)
        for current in (data, direct_data):
            self.assertIsInstance(current, module.ParameterDataManager)
            self.assertIsInstance(current, module.DataCollectorParams)
            self.assertEqual(
                (current.params.np_action, current.params.np_dynamics), (3, 3)
            )
            self.assertEqual(
                list(current.action_params.todict().keys()),
                ["alpha", "middle", "zeta"],
            )
            self.assertEqual(
                list(current.dynamics_params.todict().keys()),
                ["beta", "idle", "zulu"],
            )
            self.assertIsInstance(
                current.action_params["alpha"],
                module.ActionModelParamsDataAbstract,
            )
            self.assertIsInstance(
                current.action_params["middle"],
                module.ActionModelParamsDataAbstract,
            )
            self.assertIsInstance(
                current.dynamics_params["idle"],
                module.DynamicsParamsDataAbstract,
            )
            self.assertIsInstance(
                current.dynamics_params["zulu"],
                module.DynamicsParamsDataAbstract,
            )

        p = np.arange(1, 7, dtype=dtype)
        manager.update(data, p)
        self.assertTrue(np.array_equal(data.params.p, p))
        self.assertTrue(np.array_equal(data.action_params["alpha"].p, p[0:2]))
        self.assertTrue(np.array_equal(data.action_params["zeta"].p, p[2:3]))
        self.assertTrue(np.array_equal(data.dynamics_params["beta"].p, p[3:4]))
        self.assertTrue(np.array_equal(data.dynamics_params["zulu"].p, p[4:6]))
        self.assertTrue(np.array_equal(data.action_params["middle"].p, np.zeros(1)))
        self.assertTrue(np.array_equal(data.dynamics_params["idle"].p, np.zeros(1)))
        self.assertEqual(
            (
                action_alpha.update_calls,
                action_zeta.update_calls,
                action_middle.update_calls,
                dynamics_beta.update_calls,
                dynamics_zulu.update_calls,
                dynamics_idle.update_calls,
            ),
            (1, 1, 0, 1, 1, 0),
        )

        x = np.linspace(0.1, 0.4, 4, dtype=dtype)
        u = np.linspace(0.5, 0.6, 2, dtype=dtype)
        action_data = module.ActionModelLQR(4, 2).createData()
        dynamics_model = module.DynamicsModelAbstract(
            state, crocoddyl.DynamicsType.ContinuousControl, 0, 2
        )
        dynamics_data = dynamics_model.createData()
        dx_dp = manager.calcDiff_action(data, action_data, x, u)
        dtau_dp = manager.calcDiff_dynamics(data, dynamics_data, x, u)
        expected_action = np.tile(np.array([10, 11, 30], dtype=dtype), (state.ndx, 1))
        expected_dynamics = np.tile(np.array([40, 50, 51], dtype=dtype), (state.nv, 1))
        self.assertTrue(np.array_equal(dx_dp, expected_action))
        self.assertTrue(np.array_equal(dtau_dp, expected_dynamics))
        self.assertEqual(
            (
                action_alpha.sensitivity_calls,
                action_zeta.sensitivity_calls,
                action_middle.sensitivity_calls,
                dynamics_beta.regressor_calls,
                dynamics_zulu.regressor_calls,
                dynamics_idle.regressor_calls,
            ),
            (1, 1, 0, 1, 1, 0),
        )
        self.assertTrue(np.array_equal(manager.zero(), np.zeros(6, dtype=dtype)))
        random = manager.rand()
        self.assertEqual(random.dtype, dtype)
        self.assertTrue(np.all(random >= 0))
        self.assertTrue(np.all(random <= 1))
        self.assertIn("ParameterManager", repr(manager))

        original_item = manager.action_params["alpha"]
        manager.addParam("alpha", make_action_params(module, dtype, state, 1, 99))
        manager.removeParam("missing")
        manager.changeParamStatus("missing", True)
        self.assertFalse(manager.getParamStatus("missing"))
        self.assertIs(manager.action_params["alpha"].param, original_item.param)
        self.assertEqual(
            (manager.np, manager.np_action, manager.np_dynamics), (6, 3, 3)
        )

        manager.changeParamStatus("alpha", False)
        self.assertEqual((manager.np, manager.np_action), (4, 1))
        with self.assertRaises(crocoddyl.Exception):
            manager.update(data, np.zeros(4, dtype=dtype))
        data.resize(manager)
        manager.update(data, np.arange(1, 5, dtype=dtype))
        manager.changeParamStatus("alpha", False)
        self.assertEqual(manager.np, 4)
        manager.changeParamStatus("alpha", True)
        data.resize(manager)
        self.assertEqual((data.params.np_action, data.params.np_dynamics), (3, 3))

        data.params.p = np.ones(6, dtype=dtype)
        for item_data in data.action_params.todict().values():
            item_data.p = np.ones(item_data.np, dtype=dtype)
        for item_data in data.dynamics_params.todict().values():
            item_data.p = np.ones(item_data.np, dtype=dtype)
        data.params.active = False
        data.action_params["middle"].active = False
        data.setZero()
        self.assertTrue(np.array_equal(data.params.p, np.zeros(6, dtype=dtype)))
        self.assertFalse(data.params.active)
        self.assertFalse(data.action_params["middle"].active)
        for item_data in data.action_params.todict().values():
            self.assertTrue(np.array_equal(item_data.p, np.zeros(item_data.np)))
        for item_data in data.dynamics_params.todict().values():
            self.assertTrue(np.array_equal(item_data.p, np.zeros(item_data.np)))

        copied_item = copy.copy(original_item)
        self.assertIsNot(copied_item, original_item)
        self.assertIs(copied_item.param, original_item.param)
        copied_manager = copy.deepcopy(manager)
        self.assertIsNot(copied_manager, manager)
        self.assertIsNot(
            copied_manager.action_params["alpha"], manager.action_params["alpha"]
        )
        self.assertIs(
            copied_manager.action_params["alpha"].param,
            manager.action_params["alpha"].param,
        )
        copied_manager.changeParamStatus("alpha", False)
        self.assertFalse(copied_manager.getParamStatus("alpha"))
        self.assertTrue(manager.getParamStatus("alpha"))

        copied_data = copy.deepcopy(data)
        self.assertIsInstance(copied_data, module.ParameterDataManager)
        copied_data.params.p = np.arange(6, dtype=dtype)
        self.assertTrue(np.array_equal(data.params.p, np.arange(6, dtype=dtype)))
        copied_data.action_params["alpha"].p = np.array([8, 9], dtype=dtype)
        self.assertTrue(np.array_equal(data.action_params["alpha"].p, [8, 9]))
        manager.update(copied_data, p)

        stale = manager.createData()
        manager.removeParam("middle")
        with self.assertRaises(crocoddyl.Exception):
            stale.resize(manager)
        with self.assertRaises(crocoddyl.Exception):
            manager.update(stale, p)

        with self.assertRaises(crocoddyl.Exception):
            module.ParameterManager(None)
        with self.assertRaises(crocoddyl.Exception):
            module.ParameterItem("null", None)
        with self.assertRaises(crocoddyl.Exception):
            module.ParameterDataManager(None)
        with self.assertRaises(crocoddyl.Exception):
            manager.addParam("null", None)
        with self.assertRaises(crocoddyl.Exception):
            manager.update(manager.createData(), np.zeros(manager.np + 1, dtype=dtype))
        with self.assertRaises(crocoddyl.Exception):
            manager.addParam(
                "wrong", make_action_params(module, dtype, module.StateVector(5), 1, 1)
            )

        add_manager = module.ParameterManager(state)
        add_manager.addParam("first", make_action_params(module, dtype, state, 1, 1))
        before_add = add_manager.createData()
        add_manager.addParam("second", make_dynamics_params(module, dtype, state, 1, 2))
        with self.assertRaises(crocoddyl.Exception):
            before_add.resize(add_manager)
        with self.assertRaises(crocoddyl.Exception):
            add_manager.update(before_add, np.zeros(2, dtype=dtype))

        for alias in (
            "__getitem__",
            "np_int",
            "params_int",
            "calcDiff_int",
            "np_dyn",
            "params_dyn",
            "calcDiff_dyn",
        ):
            self.assertFalse(hasattr(manager, alias))

    def check_converter_integration(self, module, dtype):
        state = module.StateVector(4)
        manager = module.ParameterManager(state)

        class ActionProbe(module.ActionModelAbstract):
            def __init__(self):
                super().__init__(state, 2)
                self.set_calls = 0
                self.received = []

            def calc(self, data, x, u=None):
                del u
                data.xnext = x

            def calcDiff(self, data, x, u=None):
                del data, x, u

            def set_params(self, data, params):
                del data
                self.set_calls += 1
                self.received.append(params)

        action = ActionProbe()
        action_data = action.createData()
        action.set_params(action_data, manager)
        action.set_params(action_data, None)
        self.assertEqual(action.set_calls, 2)
        self.assertIs(action.received[0], manager)
        self.assertIsNone(action.received[1])

        class ActionFallback(module.ActionModelAbstract):
            def __init__(self):
                super().__init__(state, 2)

            def calc(self, data, x, u=None):
                del u
                data.xnext = x

            def calcDiff(self, data, x, u=None):
                del data, x, u

        action_fallback = ActionFallback()
        action_fallback_data = action_fallback.createData()
        action_fallback.set_params(action_fallback_data, manager)
        action_fallback.set_params(action_fallback_data, None)

        class DynamicsProbe(module.DynamicsModelAbstract):
            def __init__(self):
                super().__init__(state, crocoddyl.DynamicsType.ContinuousControl, 0, 2)
                self.set_calls = 0
                self.received = []

            def calc(self, data, x, u=None):
                del x, u
                data.vdot = np.zeros(data.vdot.shape, dtype=dtype)

            def calcDiff_xu(self, data, x, u=None):
                del data, x, u

            def set_params(self, data, params):
                del data
                self.set_calls += 1
                self.received.append(params)

        dynamics = DynamicsProbe()
        dynamics_data = dynamics.createData()
        dynamics.set_params(dynamics_data, manager)
        dynamics.set_params(dynamics_data, None)
        self.assertEqual(dynamics.set_calls, 2)
        self.assertIs(dynamics.received[0], manager)
        self.assertIsNone(dynamics.received[1])

        class DynamicsFallback(module.DynamicsModelAbstract):
            def __init__(self):
                super().__init__(state, crocoddyl.DynamicsType.ContinuousControl, 0, 2)

            def calc(self, data, x, u=None):
                del x, u
                data.vdot = np.zeros(data.vdot.shape, dtype=dtype)

            def calcDiff_xu(self, data, x, u=None):
                del data, x, u

        dynamics_fallback = DynamicsFallback()
        dynamics_fallback_data = dynamics_fallback.createData()
        with self.assertRaises(crocoddyl.Exception):
            dynamics_fallback.set_params(dynamics_fallback_data, manager)
        with self.assertRaises(crocoddyl.Exception):
            dynamics_fallback.set_params(dynamics_fallback_data, None)
        self.assertEqual(dynamics.set_calls, 2)

        for model_type in (module.ActionModelAbstract, module.DynamicsModelAbstract):
            self.assertFalse(hasattr(model_type, "set_params_py"))
            self.assertFalse(hasattr(model_type, "update_p_py"))

    def test_float64(self):
        self.check_manager(crocoddyl, np.float64)
        self.check_converter_integration(crocoddyl, np.float64)

    def test_float32(self):
        self.check_manager(crocoddyl_float32, np.float32)
        self.check_converter_integration(crocoddyl_float32, np.float32)

    def test_available_scalar_cast_path(self):
        manager64 = crocoddyl.ParameterManager(crocoddyl.StateVector(4))
        manager32 = manager64.cast(crocoddyl.DType.Float32)
        self.assertIsInstance(manager32, crocoddyl_float32.ParameterManager)
        self.assertEqual(
            (manager32.np, manager32.np_action, manager32.np_dynamics), (0, 0, 0)
        )
        roundtrip = manager32.cast(crocoddyl.DType.Float64)
        self.assertIsInstance(roundtrip, crocoddyl.ParameterManager)
        self.assertEqual(
            (roundtrip.np, roundtrip.np_action, roundtrip.np_dynamics), (0, 0, 0)
        )


if __name__ == "__main__":
    unittest.main()
