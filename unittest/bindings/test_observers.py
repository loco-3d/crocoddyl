###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################

import copy
import os
import subprocess
import sys
import textwrap
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


class ObserverBindingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state64 = crocoddyl.StateMultibody(pinocchio.buildSampleModelManipulator())
        cls.state32 = (
            cls.state64.cast(crocoddyl.DType.Float32)
            if PINOCCHIO_FLOAT32_AVAILABLE
            else None
        )
        cls.body_name = cls.state64.pinocchio.names[1]

    def scalar_cases(self):
        cast_dtype = (
            crocoddyl.DType.Float32
            if PINOCCHIO_FLOAT32_AVAILABLE
            else crocoddyl.DType.Float64
        )
        cases = [(crocoddyl, np.float64, self.state64, cast_dtype)]
        if PINOCCHIO_FLOAT32_AVAILABLE:
            cases.append(
                (
                    crocoddyl_float32,
                    np.float32,
                    self.state32,
                    crocoddyl.DType.Float64,
                )
            )
        return cases

    @staticmethod
    def state_point(state, dtype):
        dx = np.linspace(-0.08, 0.12, state.ndx, dtype=dtype)
        return np.asarray(state.integrate(state.zero(), dx), dtype=dtype)

    @staticmethod
    def make_continuous(module, dtype, state, np_=0, with_power=False):
        actuation = module.ActuationModelMultibody(state)
        implicit = module.ImplicitConstraintModelMultiple(state, actuation.nu)
        dynamics = module.DynamicsModelConstrainedForward(
            state, actuation, implicit, np_
        )
        observer_nu = state.ndx + dynamics.nu
        costs = module.CostModelSum(state, observer_nu, np_)
        state_residual = module.ResidualModelState(state, observer_nu)
        costs.addCost("state", module.CostModelResidual(state, state_residual), 0.4)
        if with_power:
            power = module.ResidualModelPower(state, observer_nu, np_)
            costs.addCost("power", module.CostModelResidual(state, power), 0.7)
        constraints = module.ConstraintModelManager(state, observer_nu, np_)
        control = module.ResidualModelControl(state, observer_nu)
        constraints.addConstraint(
            "a_running",
            module.ConstraintModelResidual(
                state,
                control,
                np.full(control.nr, -0.5, dtype=dtype),
                np.full(control.nr, 0.8, dtype=dtype),
                False,
            ),
        )
        constraints.addConstraint(
            "b_inactive",
            module.ConstraintModelResidual(
                state,
                control,
                np.full(control.nr, -1.5, dtype=dtype),
                np.full(control.nr, 1.8, dtype=dtype),
                False,
            ),
            False,
        )
        constraints.addConstraint(
            "c_terminal_bounds",
            module.ConstraintModelResidual(
                state,
                state_residual,
                np.full(state_residual.nr, -2.5, dtype=dtype),
                np.full(state_residual.nr, 2.8, dtype=dtype),
                True,
            ),
        )
        constraints.addConstraint(
            "d_terminal",
            module.ConstraintModelResidual(state, state_residual, True),
        )
        return dynamics, costs, constraints

    @staticmethod
    def make_inertial_manager(module, state, body_name):
        manager = module.ParameterManager(state)
        inertial = module.MultibodyInertialParams(
            state,
            module.LogCholeskyParametrization(),
            [body_name],
        )
        manager.addParam("inertia", inertial)
        return manager

    @staticmethod
    def control_sentinels(data, dtype, first):
        return {
            name: np.full(getattr(data, name).shape, first + i, dtype=dtype)
            for i, name in enumerate(("Fu", "Lu", "Lxu", "Luu", "Lpu", "Gu", "Hu"))
        }

    def test_abstract_wrapper_dispatch_and_fallback(self):
        for module, dtype, state, _ in self.scalar_cases():
            with self.subTest(module=module.__name__):

                class Observer(module.ObserverModelAbstract):
                    def __init__(self, bound_state=state, bound_dtype=dtype):
                        super().__init__(bound_state, 2, 3, 0, 0, 0, 0, 0, 0)
                        self.calls = []
                        self.dtype = bound_dtype

                    def calc(self, data, x, u=None):
                        self.calls.append(("calc", u is None))
                        data.xnext = x

                    def calcDiff(self, data, x, u=None):
                        self.calls.append(("calcDiff", u is None))

                    def update_p(self, data, p):
                        self.calls.append(("update_p", p.size))

                    def set_params(self, data, params):
                        self.calls.append(("set_params", params.np))
                        super().set_params(data, params)

                    def update_tau(self, tau):
                        self.calls.append(("update_tau", tau.copy()))
                        super().update_tau(tau)

                    def quasiStatic(self, data, x, maxiter=100, tol=1e-9):
                        self.calls.append(("quasiStatic", maxiter, tol))
                        return np.full(self.nu, 0.25, dtype=self.dtype)

                    def createData(self):
                        self.calls.append(("createData",))
                        return super().createData()

                model = Observer()
                data = model.createData()
                x = np.asarray(state.zero(), dtype=dtype)
                u = np.linspace(-0.2, 0.3, model.nu, dtype=dtype)
                model.calc(data, x, u)
                model.calc(data, x)
                model.calcDiff(data, x, u)
                model.calcDiff(data, x)
                model.update_p(data, np.empty(0, dtype=dtype))
                manager = module.ParameterManager(state)
                model.set_params(data, manager)
                quasi = model.quasiStatic(data, x, 17, 1e-7)
                tau = np.array([0.2, -0.4], dtype=dtype)
                model.update_tau(tau)
                np.testing.assert_array_equal(
                    quasi, np.full(model.nu, 0.25, dtype=dtype)
                )
                np.testing.assert_array_equal(model.tau_meas, tau)
                self.assertIn(("calc", False), model.calls)
                self.assertIn(("calc", True), model.calls)
                self.assertIn(("calcDiff", False), model.calls)
                self.assertIn(("calcDiff", True), model.calls)
                self.assertIn(("set_params", 0), model.calls)
                self.assertIn(("quasiStatic", 17, 1e-7), model.calls)
                self.assertIsInstance(data, module.ObserverDataAbstract)

                class FallbackObserver(module.ObserverModelAbstract):
                    def __init__(self, bound_state=state):
                        super().__init__(bound_state, 0, 0)

                    def calc(self, data, x, u=None):
                        data.xnext = x

                    def calcDiff(self, data, x, u=None):
                        pass

                    def update_p(self, data, p):
                        pass

                fallback = FallbackObserver()
                fallback_data = fallback.createData()
                fallback.set_params(fallback_data, manager)
                self.assertEqual(fallback.quasiStatic(fallback_data, x).size, 0)
                with self.assertRaises(crocoddyl.Exception):
                    module.ObserverDataAbstract(None)

    def test_euler_rk_numdiff_parameters_energy_and_terminal(self):
        for module, dtype, state, cast_dtype in self.scalar_cases():
            with self.subTest(module=module.__name__):
                manager = self.make_inertial_manager(module, state, self.body_name)
                dynamics, costs, constraints = self.make_continuous(
                    module, dtype, state, manager.np
                )
                euler = module.IntegratedObserverModelEuler(
                    dynamics, costs, constraints, 0.02
                )
                manager_data = manager.createData()
                data = euler.createData(manager_data)
                euler.set_params(data, manager)
                p = np.asarray(manager.rand(), dtype=dtype)
                euler.update_p(data, p)
                tau = np.linspace(-0.2, 0.25, euler.ntau, dtype=dtype)
                euler.update_tau(tau)
                x = self.state_point(state, dtype)
                w = np.linspace(-0.1, 0.15, euler.nu, dtype=dtype)
                euler.calc(data, x, w)
                euler.calcDiff(data, x, w)
                self.assertIsInstance(data, module.IntegratedObserverDataEuler)
                self.assertEqual(euler.nr, 0)
                self.assertEqual(data.r.size, 0)
                np.testing.assert_array_equal(
                    data.dynamics.multibody.params.p, manager_data.params.p
                )
                np.testing.assert_array_equal(euler.tau_meas, tau)
                self.assertEqual(data.Fp.shape, (state.ndx, manager.np))
                self.assertEqual(data.dE_dp.shape, (manager.np,))
                self.assertTrue(np.all(np.isfinite(data.Fp)))
                self.assertTrue(np.all(np.isfinite(data.dE_dp)))
                self.assertGreater(abs(float(data.cost)), 0.0)
                np.testing.assert_array_equal(
                    data.g,
                    np.concatenate((data.dynamics.g, data.constraints.g)),
                )
                np.testing.assert_array_equal(
                    data.h,
                    np.concatenate((data.dynamics.h, data.constraints.h)),
                )
                np.testing.assert_array_equal(
                    data.Gx,
                    np.vstack((data.dynamics.Gx, data.constraints.Gx)),
                )
                np.testing.assert_array_equal(
                    data.Hx,
                    np.vstack((data.dynamics.Hx, data.constraints.Hx)),
                )
                expected_lb = np.concatenate(
                    (
                        np.full(euler.nu, -0.5, dtype=dtype),
                        np.full(state.ndx, -2.5, dtype=dtype),
                    )
                )
                expected_ub = np.concatenate(
                    (
                        np.full(euler.nu, 0.8, dtype=dtype),
                        np.full(state.ndx, 2.8, dtype=dtype),
                    )
                )
                np.testing.assert_array_equal(euler.g_lb, expected_lb)
                np.testing.assert_array_equal(euler.g_ub, expected_ub)
                constraints.changeConstraintStatus("b_inactive", True)
                euler.calc(data, x, w)
                euler.calcDiff(data, x, w)
                np.testing.assert_array_equal(
                    euler.g_lb,
                    np.concatenate(
                        (
                            np.full(euler.nu, -0.5, dtype=dtype),
                            np.full(euler.nu, -1.5, dtype=dtype),
                            np.full(state.ndx, -2.5, dtype=dtype),
                        )
                    ),
                )
                self.assertEqual(data.g.size, euler.ng)
                constraints.changeConstraintStatus("b_inactive", False)
                euler.calc(data, x, w)
                euler.calcDiff(data, x, w)
                np.testing.assert_array_equal(euler.g_lb, expected_lb)
                self.assertEqual(
                    euler.quasiStatic(
                        data, np.asarray(state.zero(), dtype=dtype), 1
                    ).shape,
                    (euler.nu,),
                )

                numerical = module.ObserverModelNumDiff(euler, manager, False)
                numerical.disturbance = 2e-3 if dtype == np.float32 else 2e-6
                numerical_data = numerical.createData(manager.createData())
                numerical.update_tau(tau)
                numerical.update_p(numerical_data, p)
                numerical.calc(numerical_data, x, w)
                numerical.calcDiff(numerical_data, x, w)
                tolerance = 6e-2 if dtype == np.float32 else 3e-3
                for name in ("Fx", "Fu", "Fp", "Lx", "Lu", "Lp"):
                    self.assertTrue(
                        np.allclose(
                            getattr(data, name),
                            getattr(numerical_data, name),
                            atol=tolerance,
                            rtol=tolerance,
                        ),
                        (
                            name,
                            np.max(
                                np.abs(
                                    getattr(data, name) - getattr(numerical_data, name)
                                )
                            ),
                        ),
                    )
                self.assertIsNotNone(numerical_data.params_data)
                self.assertEqual(len(numerical_data.data_p), manager.np)
                self.assertEqual(
                    numerical.quasiStatic(
                        numerical_data, np.asarray(state.zero(), dtype=dtype), 1
                    ).shape,
                    (numerical.nu,),
                )
                casted_numerical = numerical.cast(cast_dtype)
                self.assertEqual(casted_numerical.np, manager.np)
                target_module = (
                    crocoddyl_float32
                    if cast_dtype == crocoddyl.DType.Float32
                    else crocoddyl
                )
                casted_dtype = (
                    np.float32 if cast_dtype == crocoddyl.DType.Float32 else np.float64
                )
                casted_numerical_data = casted_numerical.createData()
                casted_numerical.update_p(
                    casted_numerical_data, np.asarray(p, dtype=casted_dtype)
                )
                casted_numerical.calc(
                    casted_numerical_data,
                    np.asarray(x, dtype=casted_dtype),
                    np.asarray(w, dtype=casted_dtype),
                )
                casted_numerical.calcDiff(
                    casted_numerical_data,
                    np.asarray(x, dtype=casted_dtype),
                    np.asarray(w, dtype=casted_dtype),
                )
                self.assertIsInstance(
                    casted_numerical_data, target_module.ObserverDataNumDiff
                )
                np.testing.assert_allclose(
                    casted_numerical.tau_meas,
                    np.asarray(tau, dtype=casted_dtype),
                )
                numerical_sentinels = self.control_sentinels(numerical_data, dtype, 7)
                for name, value in numerical_sentinels.items():
                    setattr(numerical_data, name, value)
                numerical.calc(numerical_data, x)
                numerical.calcDiff(numerical_data, x)
                for name, value in numerical_sentinels.items():
                    np.testing.assert_array_equal(getattr(numerical_data, name), value)
                self.assertEqual(
                    numerical_data.Gp.shape, (constraints.ng_T, manager.np)
                )
                self.assertEqual(
                    numerical_data.Hp.shape, (constraints.nh_T, manager.np)
                )

                second = euler.createData(manager.createData())
                euler.set_params(second, manager)
                euler.update_p(second, p)
                euler.calc(second, x, -w)
                euler.calcDiff(second, x, -w)
                second_fx = second.Fx.copy()
                euler.calc(data, x, w)
                euler.calcDiff(data, x, w)
                np.testing.assert_array_equal(second.Fx, second_fx)
                second_numerical = numerical.createData(manager.createData())
                numerical.update_p(second_numerical, p)
                numerical.calc(second_numerical, x, -w)
                second_numerical_xnext = second_numerical.xnext.copy()
                numerical.calc(numerical_data, x, w)
                np.testing.assert_array_equal(
                    second_numerical.xnext, second_numerical_xnext
                )
                with self.assertRaises(TypeError):
                    copy.copy(data)
                with self.assertRaises(TypeError):
                    copy.copy(numerical_data)
                self.assertIsNot(copy.copy(euler), euler)
                self.assertIsNot(copy.copy(numerical), numerical)
                casted = euler.cast(cast_dtype)
                self.assertEqual(casted.np, manager.np)

                sentinels = self.control_sentinels(data, dtype, 11)
                for name, value in sentinels.items():
                    setattr(data, name, value)
                euler.calc(data, x)
                euler.calcDiff(data, x)
                for name, value in sentinels.items():
                    np.testing.assert_array_equal(getattr(data, name), value)
                self.assertEqual(data.g.shape, (constraints.ng_T,))
                self.assertEqual(data.h.shape, (constraints.nh_T,))
                np.testing.assert_array_equal(
                    euler.g_lb[: constraints.ng_T],
                    np.full(constraints.ng_T, -2.5, dtype=dtype),
                )

                for rktype, stages in (
                    (crocoddyl.RKType.two, 2),
                    (crocoddyl.RKType.three, 3),
                    (crocoddyl.RKType.four, 4),
                ):
                    rk = module.IntegratedObserverModelRK(
                        dynamics, costs, constraints, 0.02, rktype
                    )
                    rk_data = rk.createData(manager.createData())
                    rk.set_params(rk_data, manager)
                    rk.update_p(rk_data, p)
                    rk.update_tau(tau)
                    rk.calc(rk_data, x, w)
                    rk.calcDiff(rk_data, x, w)
                    self.assertEqual((rk.rktype, rk.ni), (rktype, stages))
                    self.assertEqual(rk.nr, 0)
                    self.assertEqual(rk_data.r.size, 0)
                    self.assertEqual(len(rk_data.dynamics), stages)
                    self.assertEqual(len(rk_data.costs), stages)
                    np.testing.assert_array_equal(
                        rk_data.g,
                        np.concatenate((rk_data.dynamics[0].g, rk_data.constraints.g)),
                    )
                    np.testing.assert_array_equal(
                        rk_data.h,
                        np.concatenate((rk_data.dynamics[0].h, rk_data.constraints.h)),
                    )
                    self.assertEqual(
                        rk.quasiStatic(
                            rk_data, np.asarray(state.zero(), dtype=dtype), 1
                        ).shape,
                        (rk.nu,),
                    )
                    constraints.changeConstraintStatus("b_inactive", True)
                    rk.calc(rk_data, x, w)
                    rk.calcDiff(rk_data, x, w)
                    self.assertEqual(rk_data.g.size, rk.ng)
                    constraints.changeConstraintStatus("b_inactive", False)
                    rk.calc(rk_data, x, w)
                    rk.calcDiff(rk_data, x, w)
                    np.testing.assert_array_equal(rk.g_lb, expected_lb)
                    self.assertTrue(np.all(rk_data.dissipative_E == dtype(0)))
                    self.assertTrue(np.all(rk_data.dE_dv == dtype(0)))
                    self.assertTrue(np.all(rk_data.dE_dp == dtype(0)))
                    rk_sentinels = self.control_sentinels(rk_data, dtype, 21)
                    for name, value in rk_sentinels.items():
                        setattr(rk_data, name, value)
                    rk.calc(rk_data, x)
                    rk.calcDiff(rk_data, x)
                    for name, value in rk_sentinels.items():
                        np.testing.assert_array_equal(getattr(rk_data, name), value)
                    second_rk = rk.createData(manager.createData())
                    rk.set_params(second_rk, manager)
                    rk.update_p(second_rk, p)
                    rk.calc(second_rk, x, -w)
                    second_xnext = second_rk.xnext.copy()
                    rk.calc(rk_data, x, w)
                    np.testing.assert_array_equal(second_rk.xnext, second_xnext)
                    with self.assertRaises(TypeError):
                        copy.copy(rk_data)
                    self.assertIsNot(copy.copy(rk), rk)
                    self.assertEqual(rk.cast(cast_dtype).np, manager.np)

                rk2 = module.IntegratedObserverModelRK(
                    dynamics,
                    costs,
                    constraints,
                    0.02,
                    crocoddyl.RKType.two,
                )
                rk4 = module.IntegratedObserverModelRK(
                    dynamics,
                    costs,
                    constraints,
                    0.02,
                    crocoddyl.RKType.four,
                )
                with self.assertRaises(crocoddyl.Exception):
                    rk4.calc(rk2.createData(), x, w)

                energy_dynamics, energy_costs, _ = self.make_continuous(
                    module, dtype, state, 0, True
                )
                energy_model = module.IntegratedObserverModelEuler(
                    energy_dynamics, energy_costs, None, 0.02
                )
                energy_data = energy_model.createData()
                energy_w = np.linspace(-0.12, 0.14, energy_model.nu, dtype=dtype)
                energy_model.calc(energy_data, x, energy_w)
                energy_model.calcDiff(energy_data, x, energy_w)
                self.assertTrue(np.all(np.isfinite(energy_data.dissipative_E)))
                self.assertTrue(np.all(np.isfinite(energy_data.dE_dv)))
                self.assertGreater(abs(float(energy_data.cost)), 0.0)

                with self.assertRaises(crocoddyl.Exception):
                    module.ObserverModelNumDiff(euler, None, False)
                with self.assertRaises(crocoddyl.Exception):
                    module.ObserverModelNumDiff(None)
                with self.assertRaises(crocoddyl.Exception):
                    module.ObserverDataNumDiff(None)

    def test_discretized_observer(self):
        for module, dtype, state, cast_dtype in self.scalar_cases():
            with self.subTest(module=module.__name__):
                manager = self.make_inertial_manager(module, state, self.body_name)
                implicit = module.ImplicitConstraintModelMultiple(state, 0)
                dynamics = module.DynamicsModelImpulseForward(
                    state, implicit, manager.np
                )
                costs = module.CostModelSum(state, state.ndx, manager.np)
                state_residual = module.ResidualModelState(state, state.ndx)
                costs.addCost(
                    "state",
                    module.CostModelResidual(state, state_residual),
                    0.5,
                )
                parameter_residual = module.ResidualModelParameters(
                    state, np.zeros(manager.np, dtype=dtype), state.ndx
                )
                costs.addCost(
                    "parameters",
                    module.CostModelResidual(state, parameter_residual),
                    0.3,
                )
                constraints = module.ConstraintModelManager(
                    state, state.ndx, manager.np
                )
                constraints.addConstraint(
                    "a_bounds",
                    module.ConstraintModelResidual(
                        state,
                        state_residual,
                        np.full(state.ndx, -0.7, dtype=dtype),
                        np.full(state.ndx, 1.2, dtype=dtype),
                        False,
                    ),
                )
                constraints.addConstraint(
                    "b_inactive",
                    module.ConstraintModelResidual(
                        state,
                        state_residual,
                        np.full(state.ndx, -1.7, dtype=dtype),
                        np.full(state.ndx, 2.2, dtype=dtype),
                        False,
                    ),
                    False,
                )
                constraints.addConstraint(
                    "c_parameters",
                    module.ConstraintModelResidual(state, parameter_residual, False),
                )
                constraints.addConstraint(
                    "d_terminal",
                    module.ConstraintModelResidual(state, state_residual, True),
                )
                model = module.DiscretizedObserverModel(dynamics, costs, 0, constraints)
                manager_data = manager.createData()
                data = model.createData(manager_data)
                model.set_params(data, manager)
                x = self.state_point(state, dtype)
                w = np.linspace(-0.1, 0.1, state.ndx, dtype=dtype)
                p = np.asarray(manager.rand(), dtype=dtype)
                model.update_p(data, p)
                model.calc(data, x, w)
                model.calcDiff(data, x, w)
                self.assertIsInstance(data, module.DiscretizedObserverData)
                np.testing.assert_array_equal(data.xnext, data.dynamics.vdot)
                np.testing.assert_array_equal(data.Fx, data.dynamics.Fx)
                self.assertTrue(np.all(data.Fu == dtype(0)))
                self.assertTrue(np.all(np.isfinite(data.Fp)))
                np.testing.assert_allclose(
                    data.Lp, dtype(0.3) * p, rtol=2e-5, atol=2e-5
                )
                np.testing.assert_allclose(
                    data.Hp[: manager.np],
                    np.eye(manager.np, dtype=dtype),
                    rtol=2e-5,
                    atol=2e-5,
                )
                self.assertEqual(data.g.shape, (model.ng,))
                self.assertEqual(data.h.shape, (model.nh,))
                np.testing.assert_array_equal(
                    data.g,
                    np.concatenate((data.dynamics.g, data.constraints.g)),
                )
                np.testing.assert_array_equal(
                    data.h,
                    np.concatenate((data.dynamics.h, data.constraints.h)),
                )
                np.testing.assert_array_equal(
                    model.g_lb, np.full(state.ndx, -0.7, dtype=dtype)
                )
                np.testing.assert_array_equal(
                    model.g_ub, np.full(state.ndx, 1.2, dtype=dtype)
                )
                constraints.changeConstraintStatus("b_inactive", True)
                model.calc(data, x, w)
                model.calcDiff(data, x, w)
                np.testing.assert_array_equal(
                    model.g_lb,
                    np.concatenate(
                        (
                            np.full(state.ndx, -0.7, dtype=dtype),
                            np.full(state.ndx, -1.7, dtype=dtype),
                        )
                    ),
                )
                self.assertEqual(data.g.size, model.ng)
                constraints.changeConstraintStatus("b_inactive", False)
                model.calc(data, x, w)
                model.calcDiff(data, x, w)
                self.assertTrue(
                    np.all(
                        model.quasiStatic(data, np.asarray(state.zero(), dtype=dtype))
                        == dtype(0)
                    )
                )

                numerical = module.ObserverModelNumDiff(model, manager)
                numerical_data = numerical.createData(manager.createData())
                numerical.update_p(numerical_data, p)
                numerical.calc(numerical_data, x, w)
                self.assertEqual(
                    (numerical.ng, numerical.nh, numerical.ng_T, numerical.nh_T),
                    (model.ng, model.nh, model.ng_T, model.nh_T),
                )
                self.assertEqual(
                    (
                        numerical.get_ng(),
                        numerical.get_nh(),
                        numerical.get_ng_T(),
                        numerical.get_nh_T(),
                    ),
                    (model.ng, model.nh, model.ng_T, model.nh_T),
                )
                np.testing.assert_array_equal(numerical.g_lb, model.g_lb)
                np.testing.assert_array_equal(numerical.g_ub, model.g_ub)

                sentinels = self.control_sentinels(data, dtype, 31)
                for name, value in sentinels.items():
                    setattr(data, name, value)
                data.dissipative_E = np.full_like(data.dissipative_E, 41)
                data.dE_dv = np.full_like(data.dE_dv, 42)
                data.dE_dp = np.full_like(data.dE_dp, 43)
                model.calc(data, x)
                model.calcDiff(data, x)
                for name, value in sentinels.items():
                    np.testing.assert_array_equal(getattr(data, name), value)
                self.assertEqual(data.g.shape, (model.ng_T,))
                self.assertEqual(data.h.shape, (model.nh_T,))
                self.assertTrue(np.all(data.dissipative_E == dtype(0)))
                self.assertTrue(np.all(data.dE_dv == dtype(0)))
                self.assertTrue(np.all(data.dE_dp == dtype(0)))
                target_module = (
                    crocoddyl_float32
                    if cast_dtype == crocoddyl.DType.Float32
                    else crocoddyl
                )
                casted_dtype = (
                    np.float32 if cast_dtype == crocoddyl.DType.Float32 else np.float64
                )
                casted = model.cast(cast_dtype)
                self.assertIsInstance(casted, target_module.DiscretizedObserverModel)
                self.assertIsNotNone(casted.params)
                casted_data = casted.createData()
                casted.update_p(casted_data, np.asarray(p, dtype=casted_dtype))
                casted.calc(
                    casted_data,
                    np.asarray(x, dtype=casted_dtype),
                    np.asarray(w, dtype=casted_dtype),
                )
                casted.calcDiff(
                    casted_data,
                    np.asarray(x, dtype=casted_dtype),
                    np.asarray(w, dtype=casted_dtype),
                )
                self.assertTrue(np.all(np.isfinite(casted_data.xnext)))
                second = model.createData(manager.createData())
                model.set_params(second, manager)
                model.update_p(second, p)
                model.calc(second, x, -w)
                second_xnext = second.xnext.copy()
                model.calc(data, x, w)
                np.testing.assert_array_equal(second.xnext, second_xnext)
                with self.assertRaises(TypeError):
                    copy.copy(data)
                self.assertIsNot(copy.copy(model), model)
                with self.assertRaises(crocoddyl.Exception):
                    module.DiscretizedObserverModel(None, costs, 0)
                with self.assertRaises(crocoddyl.Exception):
                    module.DiscretizedObserverData(None)
                with self.assertRaises(crocoddyl.Exception):
                    model.calc(data, x, np.zeros(state.ndx + 1, dtype=dtype))
                with self.assertRaises(crocoddyl.Exception):
                    module.DiscretizedObserverModel(dynamics, costs, 1, constraints)

    def test_generic_discrete_dynamics_dispatch(self):
        for module, dtype, _, _ in self.scalar_cases():
            with self.subTest(module=module.__name__):
                state = module.StateVector(4)

                class DiscreteDynamics(module.DynamicsModelAbstract):
                    def __init__(self, bound_state=state, bound_dtype=dtype):
                        super().__init__(
                            bound_state, crocoddyl.DynamicsType.DiscreteTime, 0, 2
                        )
                        self.calc_terminal = []
                        self.diff_terminal = []
                        self.ndx = bound_state.ndx
                        self.dtype = bound_dtype

                    def calc(self, data, x, u=None):
                        self.calc_terminal.append(u is None)
                        data.vdot = x

                    def calcDiff_xu(self, data, x, u=None):
                        self.diff_terminal.append(u is None)
                        data.Fx = np.eye(self.ndx, dtype=self.dtype)
                        data.Fu = np.zeros((self.ndx, 2), dtype=self.dtype)

                dynamics = DiscreteDynamics()
                costs = module.CostModelSum(state, state.ndx)
                model = module.DiscretizedObserverModel(dynamics, costs, 2)
                data = model.createData()
                x = np.linspace(-0.2, 0.3, state.nx, dtype=dtype)
                w = np.linspace(-0.1, 0.1, state.ndx, dtype=dtype)
                model.calc(data, x, w)
                model.calcDiff(data, x, w)
                model.calc(data, x)
                model.calcDiff(data, x)
                self.assertEqual(dynamics.calc_terminal, [False, True])
                self.assertEqual(dynamics.diff_terminal, [False, True])
                self.assertTrue(
                    np.all(
                        model.quasiStatic(data, np.zeros(state.nx, dtype=dtype))
                        == dtype(0)
                    )
                )

    def test_nested_data_python_lifetime(self):
        script = textwrap.dedent(
            """
            import gc
            import importlib
            import sys
            import numpy as np
            import pinocchio

            module = importlib.import_module(sys.argv[1])
            dtype = np.float32 if sys.argv[1].endswith("float32") else np.float64
            state64 = importlib.import_module("crocoddyl").StateMultibody(
                pinocchio.buildSampleModelManipulator()
            )
            state = state64 if dtype == np.float64 else state64.cast(
                importlib.import_module("crocoddyl").DType.Float32
            )
            actuation = module.ActuationModelMultibody(state)
            implicit = module.ImplicitConstraintModelMultiple(state, actuation.nu)
            dynamics = module.DynamicsModelConstrainedForward(
                state, actuation, implicit
            )
            nu = state.ndx + dynamics.nu
            costs = module.CostModelSum(state, nu)
            residual = module.ResidualModelCoMPosition(
                state, np.zeros(3, dtype=dtype), nu
            )
            costs.addCost("state", module.CostModelResidual(state, residual), 1.0)
            constraints = module.ConstraintModelManager(state, nu)
            x = np.asarray(state.zero(), dtype=dtype)
            u = np.zeros(nu, dtype=dtype)

            euler = module.IntegratedObserverModelEuler(
                dynamics, costs, constraints, 0.01
            )
            euler_data = euler.createData()
            euler.calc(euler_data, x, u)
            retained_costs = euler_data.costs
            retained_constraints = euler_data.constraints
            del euler_data
            gc.collect()
            costs.calc(retained_costs, x, u)
            constraints.calc(retained_constraints, x, u)

            rk = module.IntegratedObserverModelRK(
                dynamics,
                costs,
                constraints,
                0.01,
                importlib.import_module("crocoddyl").RKType.two,
            )
            rk_data = rk.createData()
            rk.calc(rk_data, x, u)
            retained_stage_cost = rk_data.costs[0]
            del rk_data
            gc.collect()
            costs.calc(retained_stage_cost, x, u)
            assert np.isfinite(retained_stage_cost.cost)
            """
        )
        module_names = ["crocoddyl"]
        if PINOCCHIO_FLOAT32_AVAILABLE:
            module_names.append("crocoddyl.float32")
        for module_name in module_names:
            with self.subTest(module=module_name):
                result = subprocess.run(
                    [sys.executable, "-c", script, module_name],
                    cwd=os.getcwd(),
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
