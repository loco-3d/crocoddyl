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
import pinocchio

try:
    import pinocchio.float32

    PINOCCHIO_FLOAT32_AVAILABLE = True
except ModuleNotFoundError:
    PINOCCHIO_FLOAT32_AVAILABLE = False

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


class ActionIntegrationBindingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state64 = crocoddyl.StateMultibody(pinocchio.buildSampleModelManipulator())
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

    def make_continuous(self, module, dtype, state):
        actuation = module.ActuationModelMultibody(state)
        implicit = module.ImplicitConstraintModelMultiple(state, actuation.nu)
        dynamics = module.DynamicsModelConstrainedForward(state, actuation, implicit)
        costs = module.CostModelSum(state, actuation.nu)
        control = module.ResidualModelControl(state, actuation.nu)
        costs.addCost("control", module.CostModelResidual(state, control), 0.7)
        constraints = module.ConstraintModelManager(state, actuation.nu)
        constraints.addConstraint(
            "running",
            module.ConstraintModelResidual(
                state,
                control,
                np.full(control.nr, -0.4, dtype=dtype),
                np.full(control.nr, 0.6, dtype=dtype),
                False,
            ),
        )
        terminal = module.ResidualModelState(state, actuation.nu)
        constraints.addConstraint(
            "terminal", module.ConstraintModelResidual(state, terminal, True)
        )
        return dynamics, costs, constraints

    def make_impulse(self, module, dtype, state):
        implicit = module.ImplicitConstraintModelMultiple(state, 0)
        frame_id = len(state.pinocchio.frames) - 1
        implicit.addConstraint(
            "contact",
            module.ContactModel(
                state,
                frame_id,
                state.pinocchio.frames[frame_id].placement,
                pinocchio.LOCAL_WORLD_ALIGNED,
                0,
                np.zeros(2, dtype=dtype),
                [True, True, True, False, False, False],
            ),
        )
        dynamics = module.DynamicsModelImpulseForward(state, implicit)
        costs = module.CostModelSum(state, 0)
        residual = module.ResidualModelState(state, 0)
        costs.addCost("state", module.CostModelResidual(state, residual), 0.7)
        constraints = module.ConstraintModelManager(state, 0)
        constraints.addConstraint(
            "terminal", module.ConstraintModelResidual(state, residual, True)
        )
        return dynamics, costs, constraints

    def state_point(self, state, dtype):
        dx = np.linspace(-0.08, 0.12, state.ndx, dtype=dtype)
        return np.asarray(state.integrate(state.zero(), dx), dtype=dtype)

    def test_dynamics_euler_rk_parameters_copy_and_terminal(self):
        for module, dtype, state in self.scalar_cases():
            with self.subTest(module=module.__name__):
                dynamics, costs, constraints = self.make_continuous(
                    module, dtype, state
                )
                time = module.IntegratorTime(0.02, True)
                euler = module.IntegratedActionModelEuler(
                    dynamics, costs, constraints, None, time
                )
                self.assertIs(euler.dynamics, dynamics)
                self.assertIs(euler.costs, costs)
                self.assertIs(euler.constraints, constraints)
                self.assertIs(euler.integrator_time, time)

                x = self.state_point(state, dtype)
                u = np.linspace(-0.15, 0.2, euler.nu, dtype=dtype)
                data = euler.createData()
                self.assertIsInstance(data, module.IntegratedActionDataEuler)
                self.assertIsInstance(data.dynamics, module.DynamicsDataAbstract)
                euler.calc(data, x, u)
                euler.calcDiff(data, x, u)
                self.assertEqual(data.Fx.shape, (state.ndx, state.ndx))
                self.assertEqual(data.Fu.shape, (state.ndx, euler.nu))
                self.assertEqual(data.g.shape, (constraints.ng,))
                self.assertEqual(data.h.shape, (constraints.nh,))

                numerical = module.ActionModelNumDiff(euler)
                numerical.disturbance = float(2e-3 if dtype == np.float32 else 1e-6)
                numerical_data = numerical.createData()
                numerical.calc(numerical_data, x, u)
                numerical.calcDiff(numerical_data, x, u)
                tolerance = 4e-2 if dtype == np.float32 else 2e-4
                self.assertTrue(np.allclose(data.Fx, numerical_data.Fx, atol=tolerance))
                self.assertTrue(np.allclose(data.Fu, numerical_data.Fu, atol=tolerance))

                time.timeStep = 0.037
                self.assertAlmostEqual(euler.dt, 0.037, places=5)
                self.assertIn("0.037", repr(euler))
                cast_dtype = (
                    crocoddyl.DType.Float32
                    if dtype == np.float64
                    else crocoddyl.DType.Float64
                )
                if PINOCCHIO_FLOAT32_AVAILABLE:
                    live_cast = euler.cast(cast_dtype)
                    self.assertAlmostEqual(live_cast.dt, 0.037, places=5)
                time.timeStep = 0.02

                manager = module.ParameterManager(state)
                manager.addParam("time", module.IntegratorTimeoptParams(state, time))
                euler.set_params(data, manager)
                p = np.array([np.log(0.025)], dtype=dtype)
                euler.update_p(data, p)
                euler.calc(data, x, u)
                euler.calcDiff(data, x, u)
                self.assertEqual(euler.np, 1)
                self.assertEqual(data.Fp.shape, (state.ndx,))
                self.assertEqual(data.Lpx.shape, (state.ndx,))
                self.assertEqual(data.Lpu.shape, (euler.nu,))
                self.assertIsNotNone(data.params)
                self.assertTrue(np.all(np.isfinite(data.Fp)))

                copied_data = copy.copy(data)
                copied_data.Fx = np.zeros_like(copied_data.Fx)
                self.assertFalse(np.array_equal(copied_data.Fx, data.Fx))
                copied_model = copy.copy(euler)
                self.assertIs(copied_model.integrator_time, time)

                sentinels = {
                    "Fu": np.full(data.Fu.shape, 11, dtype=dtype),
                    "Lu": np.full(data.Lu.shape, 12, dtype=dtype),
                    "Lxu": np.full(data.Lxu.shape, 13, dtype=dtype),
                    "Luu": np.full(data.Luu.shape, 14, dtype=dtype),
                    "Lpu": np.full(data.Lpu.shape, 15, dtype=dtype),
                    "Gu": np.full(data.Gu.shape, 16, dtype=dtype),
                    "Hu": np.full(data.Hu.shape, 17, dtype=dtype),
                }
                for name, value in sentinels.items():
                    setattr(data, name, value)
                euler.calc(data, x)
                euler.calcDiff(data, x)
                for name, value in sentinels.items():
                    self.assertTrue(np.array_equal(getattr(data, name), value))
                self.assertTrue(np.array_equal(data.xnext, x))
                self.assertEqual(data.g.shape, (constraints.ng_T,))
                self.assertEqual(data.h.shape, (constraints.nh_T,))
                with self.assertRaises(crocoddyl.Exception):
                    euler.calc(data, x, np.zeros(euler.nu + 1, dtype=dtype))

                rk_models = []
                for rk_type, stages in (
                    (crocoddyl.RKType.two, 2),
                    (crocoddyl.RKType.three, 3),
                    (crocoddyl.RKType.four, 4),
                ):
                    rk = module.IntegratedActionModelRK(
                        dynamics, costs, constraints, None, time, rk_type
                    )
                    rk_data = rk.createData()
                    rk_models.append((rk, rk_data))
                    rk.calc(rk_data, x, u)
                    rk.calcDiff(rk_data, x, u)
                    self.assertEqual(rk.ni, stages)
                    self.assertEqual(len(rk_data.dynamics), stages)
                    self.assertTrue(np.all(np.isfinite(rk_data.Fx)))
                    rk_sentinels = {
                        "Fu": np.full(rk_data.Fu.shape, 21, dtype=dtype),
                        "Lu": np.full(rk_data.Lu.shape, 22, dtype=dtype),
                        "Lxu": np.full(rk_data.Lxu.shape, 23, dtype=dtype),
                        "Luu": np.full(rk_data.Luu.shape, 24, dtype=dtype),
                        "Lpu": np.full(rk_data.Lpu.shape, 25, dtype=dtype),
                        "Gu": np.full(rk_data.Gu.shape, 26, dtype=dtype),
                        "Hu": np.full(rk_data.Hu.shape, 27, dtype=dtype),
                    }
                    for name, value in rk_sentinels.items():
                        setattr(rk_data, name, value)
                    rk.calc(rk_data, x)
                    rk.calcDiff(rk_data, x)
                    for name, value in rk_sentinels.items():
                        self.assertTrue(np.array_equal(getattr(rk_data, name), value))
                    self.assertTrue(np.array_equal(rk_data.xnext, x))

                with self.assertRaises(crocoddyl.Exception):
                    rk_models[-1][0].calc(rk_models[0][1], x, u)
                with self.assertRaises(crocoddyl.Exception):
                    rk_models[-1][0].calcDiff(rk_models[0][1], x, u)

                impulse, impulse_costs, _ = self.make_impulse(module, dtype, state)
                with self.assertRaises(crocoddyl.Exception):
                    module.IntegratedActionModelEuler(impulse, impulse_costs)
                with self.assertRaises(crocoddyl.Exception):
                    module.IntegratedActionModelEuler(None, costs)
                with self.assertRaises(crocoddyl.Exception):
                    module.IntegratedActionDataAbstract(None)
                with self.assertRaises(crocoddyl.Exception):
                    module.IntegratedActionDataEuler(None)
                with self.assertRaises(crocoddyl.Exception):
                    module.IntegratedActionDataRK(None)

        if PINOCCHIO_FLOAT32_AVAILABLE:
            casted = crocoddyl.IntegratedActionModelEuler(
                *self.make_continuous(crocoddyl, np.float64, self.state64)
            ).cast(crocoddyl.DType.Float32)
            self.assertIsInstance(casted, crocoddyl_float32.IntegratedActionModelEuler)

    def test_discretized_direct_dynamics_and_errors(self):
        for module, dtype, state in self.scalar_cases():
            with self.subTest(module=module.__name__):
                dynamics, costs, constraints = self.make_impulse(module, dtype, state)
                model = module.DiscretizedActionModel(dynamics, costs, constraints)
                data = model.createData()
                self.assertIsInstance(data, module.DiscretizedActionData)
                self.assertIs(model.dynamics, dynamics)
                self.assertIs(model.costs, costs)
                self.assertIs(model.constraints, constraints)
                x = self.state_point(state, dtype)
                u = np.empty(0, dtype=dtype)
                model.calc(data, x, u)
                self.assertTrue(np.array_equal(data.xnext, data.dynamics.vdot))
                self.assertEqual(data.cost, data.costs.cost)
                model.calcDiff(data, x, u)
                self.assertTrue(np.array_equal(data.Fx, data.dynamics.Fx))
                self.assertTrue(np.array_equal(data.Fu, data.dynamics.Fu))
                self.assertTrue(np.array_equal(data.Lx, data.costs.Lx))
                sentinels = {
                    "Fu": np.full(data.Fu.shape, 31, dtype=dtype),
                    "Lu": np.full(data.Lu.shape, 32, dtype=dtype),
                    "Lxu": np.full(data.Lxu.shape, 33, dtype=dtype),
                    "Luu": np.full(data.Luu.shape, 34, dtype=dtype),
                    "Lpu": np.full(data.Lpu.shape, 35, dtype=dtype),
                    "Gu": np.full(data.Gu.shape, 36, dtype=dtype),
                    "Hu": np.full(data.Hu.shape, 37, dtype=dtype),
                }
                for name, value in sentinels.items():
                    setattr(data, name, value)
                model.calc(data, x)
                model.calcDiff(data, x)
                for name, value in sentinels.items():
                    self.assertTrue(np.array_equal(getattr(data, name), value))
                self.assertTrue(np.array_equal(data.xnext, x))
                self.assertTrue(np.array_equal(data.Fx, np.eye(state.ndx, dtype=dtype)))
                copied = copy.copy(model)
                copied_data = copied.createData()
                copied.calc(copied_data, x, u)
                self.assertTrue(
                    np.array_equal(copied_data.xnext, copied_data.dynamics.vdot)
                )

                continuous, continuous_costs, _ = self.make_continuous(
                    module, dtype, state
                )
                with self.assertRaises(crocoddyl.Exception):
                    module.DiscretizedActionModel(continuous, continuous_costs)
                with self.assertRaises(crocoddyl.Exception):
                    module.DiscretizedActionModel(None, costs)
                with self.assertRaises(crocoddyl.Exception):
                    module.DiscretizedActionData(None)

    def test_python_terminal_override_receives_no_control(self):
        for module, dtype, state in self.scalar_cases():
            with self.subTest(module=module.__name__):

                class IntegratedDynamics(module.IntegratedActionModelAbstract):
                    def __init__(self, dynamics, costs, constraints):
                        super().__init__(dynamics, costs, constraints)
                        self.calc_u = "unset"
                        self.calc_diff_u = "unset"

                    def calc(self, data, x, u=None):
                        self.calc_u = u

                    def calcDiff(self, data, x, u=None):
                        self.calc_diff_u = u

                dynamics, costs, constraints = self.make_continuous(
                    module, dtype, state
                )
                model = IntegratedDynamics(dynamics, costs, constraints)
                data = model.createData()
                x = np.zeros(state.nx, dtype=dtype)
                u = np.zeros(model.nu, dtype=dtype)
                model.calc(data, x, u)
                self.assertIsNotNone(model.calc_u)
                model.calc(data, x)
                self.assertIsNone(model.calc_u)
                model.calcDiff(data, x, u)
                self.assertIsNotNone(model.calc_diff_u)
                model.calcDiff(data, x)
                self.assertIsNone(model.calc_diff_u)
                self.assertIs(model.dynamics, dynamics)
                self.assertIs(model.costs, costs)
                self.assertIs(model.constraints, constraints)


if __name__ == "__main__":
    unittest.main()
