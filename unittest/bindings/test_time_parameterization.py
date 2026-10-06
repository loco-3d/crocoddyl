###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################

import copy
import importlib
import unittest

import example_robot_data
import numpy as np

try:
    importlib.import_module("pinocchio.float32")

    PINOCCHIO_FLOAT32_AVAILABLE = True
except ModuleNotFoundError:
    PINOCCHIO_FLOAT32_AVAILABLE = False

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


class TimeParameterizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        robot = example_robot_data.load("talos_arm")
        cls.state64 = crocoddyl.StateMultibody(robot.model)
        cls.state32 = (
            cls.state64.cast(crocoddyl.DType.Float32)
            if PINOCCHIO_FLOAT32_AVAILABLE
            else None
        )

    def scalar_cases(self):
        cases = [(crocoddyl, np.float64, self.state64)]
        if PINOCCHIO_FLOAT32_AVAILABLE:
            cases.append((crocoddyl_float32, np.float32, self.state32))
        return cases

    def test_integrator_time_construction_sync_copy_and_errors(self):
        for module, dtype, _ in self.scalar_cases():
            with self.subTest(module=module.__name__):
                default_time = module.IntegratorTime()
                self.assertAlmostEqual(default_time.timeStep, 1e-3, places=6)
                self.assertAlmostEqual(default_time.timeStep2, 1e-6, places=8)
                self.assertFalse(default_time.timeopt)

                time = module.IntegratorTime(0.0, True)
                self.assertEqual(time.timeStep, 0.0)
                self.assertEqual(time.timeStep2, 0.0)
                self.assertTrue(time.timeopt)
                time.timeStep = 0.2
                time.timeopt = False
                self.assertAlmostEqual(time.timeStep2, 0.04, places=6)
                self.assertFalse(time.timeopt)
                with self.assertRaises(crocoddyl.Exception):
                    time.timeStep = -1e-3
                with self.assertRaises(crocoddyl.Exception):
                    module.IntegratorTime(-1e-3)
                with self.assertRaises(AttributeError):
                    time.timeStep2 = 1.0

                copied = copy.copy(time)
                copied.timeStep = 0.3
                self.assertAlmostEqual(time.timeStep, 0.2, places=6)
                self.assertAlmostEqual(copied.timeStep2, 0.09, places=6)
                self.assertIn("IntegratorTime", repr(time))

        casted = crocoddyl.IntegratorTime(0.3, True).cast(crocoddyl.DType.Float32)
        self.assertIsInstance(casted, crocoddyl_float32.IntegratorTime)
        self.assertAlmostEqual(casted.timeStep, 0.3, places=6)
        self.assertTrue(casted.timeopt)

    def test_time_parameter_model_data_manager_copy_cast_and_failures(self):
        for module, dtype, state in self.scalar_cases():
            with self.subTest(module=module.__name__):
                time = module.IntegratorTime(0.02, True)
                model = module.IntegratorTimeoptParams(state, time)
                self.assertEqual(model.np, 1)
                self.assertIs(model.integrator_time, time)

                data = model.createData()
                self.assertIsInstance(data, module.IntegratorTimeoptParamsData)
                self.assertIsInstance(data, module.ActionModelParamsDataAbstract)
                self.assertIsInstance(data, module.ParamsDataAbstract)
                self.assertEqual((data.np, data.np_action, data.np_dynamics), (1, 1, 0))
                self.assertEqual(data.p.shape, (1,))
                self.assertTrue(model.checkData(data))

                p = np.array([np.log(0.03)], dtype=dtype)
                model.update(data, p)
                self.assertTrue(np.array_equal(data.p, p))
                self.assertAlmostEqual(data.dt, 0.03, places=5)
                self.assertAlmostEqual(data.dt_dp, 0.03, places=5)
                self.assertAlmostEqual(time.timeStep, 0.03, places=5)
                data.dt = 0.4
                data.dt_dp = 0.5
                data.active = False
                data.resize(1, 0)
                self.assertFalse(data.active)
                self.assertEqual(data.dt, 0.0)
                self.assertEqual(data.dt_dp, 0.0)

                model.lb = np.array([-8.0], dtype=dtype)
                model.ub = np.array([-2.0], dtype=dtype)
                self.assertTrue(np.array_equal(model.lb, [-8.0]))
                self.assertTrue(np.array_equal(model.ub, [-2.0]))
                with self.assertRaises(crocoddyl.Exception):
                    model.update(data, np.zeros(2, dtype=dtype))
                with self.assertRaises(crocoddyl.Exception):
                    model.update(module.ParamsDataAbstract(1, 0), p)

                collector = module.DataCollectorParams(data)
                self.assertIs(collector.params, data)
                copied_data = copy.copy(data)
                copied_data.p = np.ones(1, dtype=dtype)
                self.assertFalse(np.array_equal(copied_data.p, data.p))
                copied_model = copy.copy(model)
                copied_model.update(
                    copied_model.createData(),
                    np.array([np.log(0.04)], dtype=dtype),
                )
                self.assertAlmostEqual(time.timeStep, 0.04, places=5)

                manager = module.ParameterManager(state)
                manager.addParam("time", model)
                self.assertEqual(
                    (manager.np, manager.np_action, manager.np_dynamics),
                    (1, 1, 0),
                )
                manager_data = manager.createData()
                manager.update(manager_data, p)
                self.assertAlmostEqual(time.timeStep, 0.03, places=5)
                manager.changeParamStatus("time", False)
                self.assertEqual((manager.np, manager.np_action), (0, 0))
                manager_data = manager.createData()
                manager.update(manager_data, np.zeros(0, dtype=dtype))
                self.assertAlmostEqual(time.timeStep, 0.03, places=5)

                with self.assertRaises(crocoddyl.Exception):
                    module.IntegratorTimeoptParams(None, time)
                with self.assertRaises(crocoddyl.Exception):
                    module.IntegratorTimeoptParams(state, None)
                with self.assertRaises(crocoddyl.Exception):
                    module.IntegratorTimeoptParams(
                        state, module.IntegratorTime(0.01, False)
                    )
                with self.assertRaises(crocoddyl.Exception):
                    module.IntegratorTimeoptParams(module.StateVector(3), time)
                with self.assertRaises(crocoddyl.Exception):
                    module.IntegratorTimeoptParamsData(None)

        if PINOCCHIO_FLOAT32_AVAILABLE:
            model64 = crocoddyl.IntegratorTimeoptParams(
                self.state64, crocoddyl.IntegratorTime(0.02, True)
            )
            model64.lb = np.array([-8.0])
            model64.ub = np.array([-2.0])
            model32 = model64.cast(crocoddyl.DType.Float32)
            self.assertIsInstance(model32, crocoddyl_float32.IntegratorTimeoptParams)
            self.assertEqual(model32.state.nx, model64.state.nx)
            self.assertTrue(np.allclose(model32.lb, model64.lb))
            self.assertTrue(np.allclose(model32.ub, model64.ub))
            self.assertIsNot(model32.integrator_time, model64.integrator_time)

    def test_existing_integrators_refresh_shared_time_and_keep_terminal_route(self):
        for module, dtype, state in self.scalar_cases():
            with self.subTest(module=module.__name__):
                actuation = module.ActuationModelMultibody(state)
                costs = module.CostModelSum(state, actuation.nu)
                implicit = module.ImplicitConstraintModelMultiple(state, actuation.nu)
                dynamics = module.DynamicsModelConstrainedForward(
                    state, actuation, implicit
                )
                constraints = module.ConstraintModelManager(state, actuation.nu)
                time = module.IntegratorTime(0.02)
                euler = module.IntegratedActionModelEuler(
                    dynamics, costs, constraints, None, time
                )
                euler_shared = copy.copy(euler)
                euler.integrator_time.timeopt = True
                params = module.IntegratorTimeoptParams(state, euler.integrator_time)
                params.update(
                    params.createData(),
                    np.array([np.log(0.03)], dtype=dtype),
                )
                self.assertAlmostEqual(euler.dt, 0.03, places=5)

                data = euler.createData()
                x = np.asarray(state.zero(), dtype=dtype)
                u = np.zeros(euler.nu, dtype=dtype)
                euler.calc(data, x, u)
                self.assertAlmostEqual(euler.dt, 0.03, places=5)
                shared_data = euler_shared.createData()
                euler_shared.calc(shared_data, x, u)
                self.assertAlmostEqual(euler_shared.dt, 0.03, places=5)
                euler.integrator_time.timeStep = 0.04
                euler.calc(data, x, u)
                self.assertAlmostEqual(euler.dt, 0.04, places=5)
                euler_shared.calc(shared_data, x, u)
                self.assertAlmostEqual(euler_shared.dt, 0.04, places=5)
                euler.calc(data, x)
                self.assertTrue(np.array_equal(data.xnext, x))

                with self.assertRaises(TypeError):
                    module.IntegratorTimeoptParams(euler)

                rk_time = module.IntegratorTime(0.02)
                rk = module.IntegratedActionModelRK(
                    dynamics,
                    costs,
                    constraints,
                    None,
                    rk_time,
                    crocoddyl.RKType.two,
                )
                rk.integrator_time.timeopt = True
                rk_params = module.IntegratorTimeoptParams(state, rk.integrator_time)
                rk_params.update(
                    rk_params.createData(),
                    np.array([np.log(0.03)], dtype=dtype),
                )
                rk_data = rk.createData()
                rk.calc(rk_data, x, u)
                self.assertAlmostEqual(rk.dt, 0.03, places=5)
                rk.integrator_time.timeStep = 0.04
                rk.calc(rk_data, x, u)
                self.assertAlmostEqual(rk.dt, 0.04, places=5)
                rk.calc(rk_data, x)
                self.assertTrue(np.array_equal(rk_data.xnext, x))


if __name__ == "__main__":
    unittest.main()
