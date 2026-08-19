###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026-2026, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################

import copy
import os
import subprocess
import sys
import unittest

import numpy as np

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


def make_dynamics_params(module, dtype, state, np_):
    class DynamicsParams(module.DynamicsParamsAbstract):
        def __init__(self):
            super().__init__(state, np_)
            self.update_calls = 0
            self.current_p = np.zeros(np_, dtype=dtype)

        def update(self, data, p):
            self.update_calls += 1
            self.current_p = np.array(p, dtype=dtype)
            data.p = self.current_p

        def computeJointTorqueRegressor(self, data, params, x, u):
            del data, params, x, u
            return np.zeros((state.nv, np_), dtype=dtype)

    item = DynamicsParams()
    manager = module.ParameterManager(state)
    manager.addParam("dynamics", item)
    return manager, item


def make_observer(module, dtype, state, np_, params_item):
    class DiscreteDynamics(module.DynamicsModelAbstract):
        def __init__(self):
            super().__init__(state, crocoddyl.DynamicsType.DiscreteTime, np_, 1)

        def calc(self, data, x, u=None):
            value = np.array(x, dtype=dtype, copy=True)
            value[0] += self.tau_meas[0]
            for j in range(np_):
                value += (j + 1) * params_item.current_p[j]
            data.vdot = value

        def calcDiff_xu(self, data, x, u=None):
            data.Fx = np.eye(state.ndx, dtype=dtype)
            data.Fu = np.zeros(data.Fu.shape, dtype=dtype)

        def calcDiff_p(self, data, x, u):
            data.Fp = np.tile(np.arange(1, np_ + 1, dtype=dtype), (state.ndx, 1))

        def set_params(self, data, manager):
            if manager.np != np_:
                raise ValueError("inconsistent parameter dimension")

        def update_p(self, data, p):
            if len(p) != np_:
                raise ValueError("inconsistent parameter dimension")

    dynamics = DiscreteDynamics()
    costs = module.CostModelSum(state, state.ndx, np_)
    costs.addCost(
        "control",
        module.CostModelResidual(state, module.ResidualModelControl(state, state.ndx)),
        1.0,
    )
    return module.DiscretizedObserverModel(dynamics, costs, 1), dynamics


def make_parameter_constraints(module, dtype, state, nu, np_):
    constraints = module.ConstraintModelManager(state, nu, np_)
    constraints.addConstraint(
        "a_control",
        module.ConstraintModelResidual(state, module.ResidualModelControl(state, nu)),
    )
    constraints.addConstraint(
        "b_parameter",
        module.ConstraintModelResidual(
            state,
            module.ResidualModelParameters(state, np.zeros(np_, dtype=dtype), nu),
        ),
    )
    return constraints


class ParameterizedProblemsTest(unittest.TestCase):
    def check_parametrized_shooting(self, module, dtype):
        phase0_a = module.ActionModelLQR(4, 2, 1, 1, 1)
        phase0_b = module.ActionModelLQR(4, 2, 1, 1, 1)
        phase1 = module.ActionModelLQR(4, 1, 2, 1, 1)
        terminal = module.ActionModelLQR(4, 1, 2, 1, 1)
        params0 = module.ParameterManager(phase0_a.state)
        params0.addParam("lqr", module.LQRParams(phase0_a.state, 1))
        params1 = module.ParameterManager(phase1.state)
        params1.addParam("lqr", module.LQRParams(phase1.state, 2))
        constraints0 = make_parameter_constraints(module, dtype, phase0_a.state, 2, 1)
        constraints1 = make_parameter_constraints(module, dtype, phase1.state, 1, 2)
        x0 = np.linspace(-0.3, 0.3, 4, dtype=dtype)
        problem = module.ShootingProblem(
            x0=x0,
            modelPhases=[[phase0_a, phase0_b], [phase1]],
            terminalModel=terminal,
            paramsModel=[params0, params1],
            paramsConstraints=[constraints0, constraints1],
        )

        self.assertIsInstance(problem, module.ShootingProblem)
        self.assertIsInstance(problem, module.ProblemAbstract)
        self.assertEqual((problem.T, problem.n_phases), (3, 2))
        self.assertEqual(list(problem.phase_idxs), [0, 2])
        self.assertEqual(list(problem.phase_edxs), [2, 3])
        self.assertEqual(len(problem.runningPhaseModels[0]), 2)
        self.assertEqual(len(problem.runningPhaseDatas[1]), 1)
        self.assertIs(problem.runningPhaseModels[0][0], problem.runningModels[0])
        self.assertIs(problem.runningPhaseModels[1][0], problem.runningModels[2])
        problem.runningPhaseDatas[0][0].cost = 1.0
        problem.runningPhaseDatas[1][0].cost = 2.0
        self.assertEqual(problem.runningDatas[0].cost, 1.0)
        self.assertEqual(problem.runningDatas[2].cost, 2.0)
        self.assertTrue(problem.hasParamsConstraints)
        self.assertIs(problem.paramsModel[0], params0)
        self.assertIs(problem.paramsModel[1], params1)

        p0 = np.array([0.25], dtype=dtype)
        p1 = np.array([-0.2, 0.4], dtype=dtype)
        problem.update_p(p0, 0)
        problem.update_p(p1, 1)
        self.assertTrue(np.array_equal(problem.paramsData[0].params.p, p0))
        self.assertTrue(np.array_equal(problem.paramsData[1].params.p, p1))
        us = [
            np.full(2, 0.1, dtype=dtype),
            np.full(2, -0.2, dtype=dtype),
            np.full(1, 0.3, dtype=dtype),
        ]
        xs = problem.rollout(us)
        cost = problem.calc(xs, us)
        self.assertTrue(np.isfinite(cost))
        self.assertEqual(problem.calcDiff(xs, us), cost)
        self.assertEqual(problem.runningDatas[0].Fp.size, 4)
        self.assertEqual(problem.runningDatas[2].Fp.size, 8)
        self.assertEqual(problem.terminalData.Fp.size, 8)

        constraints1.calc(problem.paramsConstraintsData[1], xs[2], us[2])
        constraints1.calcDiff(problem.paramsConstraintsData[1], xs[2], us[2])
        self.assertTrue(
            np.allclose(problem.paramsConstraintsData[1].h[: len(us[2])], us[2])
        )
        self.assertTrue(
            np.allclose(problem.paramsConstraintsData[1].h[len(us[2]) :], p1)
        )

        original_models = list(problem.runningModels)
        original_datas = list(problem.runningDatas)
        original_terminal_model = problem.terminalModel
        original_terminal_data = problem.terminalData
        original_phase_idxs = list(problem.phase_idxs)
        original_phase_edxs = list(problem.phase_edxs)
        original_params = list(problem.paramsModel)
        original_params_data = [data.params for data in problem.paramsData]
        original_constraints = list(problem.paramsConstraints)
        original_constraints_data = list(problem.paramsConstraintsData)

        def check_structural_identity():
            for current, original in zip(problem.runningModels, original_models):
                self.assertIs(current, original)
            self.assertIs(problem.terminalModel, original_terminal_model)
            self.assertEqual(list(problem.phase_idxs), original_phase_idxs)
            self.assertEqual(list(problem.phase_edxs), original_phase_edxs)
            for current, original in zip(problem.paramsModel, original_params):
                self.assertIs(current, original)
            for current, original in zip(
                problem.paramsConstraints, original_constraints
            ):
                self.assertIs(current, original)
            for i, original in enumerate(original_datas):
                original.cost = float(i + 1)
                self.assertEqual(problem.runningDatas[i].cost, float(i + 1))
            original_terminal_data.cost = 7.0
            self.assertEqual(problem.terminalData.cost, 7.0)
            for i, original in enumerate(original_constraints_data):
                original.h = np.full(original.h.shape, dtype(i + 2), dtype=dtype)
                self.assertTrue(
                    np.array_equal(problem.paramsConstraintsData[i].h, original.h)
                )
            problem.update_p(p0, 0)
            problem.update_p(p1, 1)
            self.assertTrue(np.array_equal(original_params_data[0].p, p0))
            self.assertTrue(np.array_equal(original_params_data[1].p, p1))
            current_xs = problem.rollout(us)
            current_cost = problem.calc(current_xs, us)
            self.assertTrue(np.isfinite(current_cost))
            self.assertEqual(problem.calcDiff(current_xs, us), current_cost)

        rejected_mutations = (
            lambda: problem.circularAppend(phase0_a, original_datas[0]),
            lambda: problem.circularAppend(phase0_a),
            lambda: problem.updateNode(0, phase0_a, original_datas[0]),
            lambda: problem.updateModel(0, phase0_a),
            lambda: setattr(problem, "runningModels", original_models),
            lambda: setattr(problem, "terminalModel", terminal),
        )
        for mutate in rejected_mutations:
            with self.assertRaisesRegex(Exception, "must be reconstructed"):
                mutate()
            check_structural_identity()

        base_descriptor_mutations = (
            lambda: module.ShootingProblem.circularAppend(
                problem, phase0_a, original_datas[0]
            ),
            lambda: module.ShootingProblem.circularAppend(problem, phase0_a),
            lambda: module.ShootingProblem.updateNode(
                problem, 0, phase0_a, original_datas[0]
            ),
            lambda: module.ShootingProblem.updateModel(problem, 0, phase0_a),
            lambda: module.ShootingProblem.runningModels.__set__(
                problem, original_models
            ),
            lambda: module.ShootingProblem.terminalModel.__set__(problem, terminal),
        )
        for mutate in base_descriptor_mutations:
            with self.assertRaisesRegex(Exception, "must be reconstructed"):
                mutate()
            check_structural_identity()

        modified_x0 = x0 + dtype(0.1)
        problem.x0 = modified_x0
        self.assertTrue(np.array_equal(problem.x0, modified_x0))
        problem.x0 = x0
        problem.is_updated = True
        self.assertTrue(problem.is_updated)
        self.assertFalse(problem.is_updated)
        check_structural_identity()

        shallow = copy.copy(problem)
        deep = copy.deepcopy(problem)
        shallow.runningDatas[0].cost = 7.0
        self.assertEqual(problem.runningDatas[0].cost, 7.0)
        deep.update_p(p1 / dtype(2), 1)
        self.assertTrue(np.array_equal(problem.paramsData[1].params.p, p1 / dtype(2)))
        problem.update_p(p1, 1)
        self.assertEqual(shallow.calc(xs, us), cost)

        with self.assertRaises(crocoddyl.Exception):
            problem.update_p(p0, 2)
        with self.assertRaises(IndexError):
            problem.runningPhaseModels[2]
        with self.assertRaises(crocoddyl.Exception):
            problem.calc(xs.tolist()[:-1], us)
        with self.assertRaises(crocoddyl.Exception):
            module.ShootingProblem(x0, [], terminal, params0)
        with self.assertRaises(crocoddyl.Exception):
            module.ShootingProblem(x0, [phase0_a], None, params0)
        with self.assertRaises(crocoddyl.Exception):
            module.ShootingProblem(x0, [phase1], phase1, params0)
        with self.assertRaises(crocoddyl.Exception):
            module.ShootingProblem(
                x0,
                [phase0_a],
                phase0_a,
                params0,
                make_parameter_constraints(module, dtype, phase0_a.state, 1, 1),
            )

        params0.addParam("inactive", module.LQRParams(phase0_a.state, 1), False)
        params0.changeParamStatus("inactive", True)
        with self.assertRaises(crocoddyl.Exception):
            problem.update_p(np.zeros(2, dtype=dtype), 0)

    def check_ordinary_shooting_mutations(self, module, dtype):
        model = module.ActionModelLQR(4, 2)
        replacement = module.ActionModelLQR(4, 2)
        problem = module.ShootingProblem(
            np.zeros(4, dtype=dtype), [model, model], model
        )
        replacement_data = replacement.createData()

        problem.circularAppend(replacement, replacement_data)
        self.assertIs(problem.runningModels[-1], replacement)
        replacement_data.cost = 1.0
        self.assertEqual(problem.runningDatas[-1].cost, 1.0)
        problem.circularAppend(model)
        self.assertIs(problem.runningModels[-1], model)
        problem.updateNode(0, replacement, replacement_data)
        self.assertIs(problem.runningModels[0], replacement)
        replacement_data.cost = 2.0
        self.assertEqual(problem.runningDatas[0].cost, 2.0)
        problem.updateModel(1, replacement)
        self.assertIs(problem.runningModels[1], replacement)
        problem.runningModels = [model, model]
        self.assertTrue(all(current is model for current in problem.runningModels))
        problem.terminalModel = replacement
        self.assertIs(problem.terminalModel, replacement)

    def check_observation_problem(self, module, dtype):
        state = module.StateVector(4)
        params0, item0 = make_dynamics_params(module, dtype, state, 1)
        params1, item1 = make_dynamics_params(module, dtype, state, 2)
        observer0a, dynamics0a = make_observer(module, dtype, state, 1, item0)
        observer0b, dynamics0b = make_observer(module, dtype, state, 1, item0)
        observer1, dynamics1 = make_observer(module, dtype, state, 2, item1)
        terminal, terminal_dynamics = make_observer(module, dtype, state, 2, item1)
        self.assertEqual(
            [
                dynamics0a.np,
                dynamics0b.np,
                dynamics1.np,
                terminal_dynamics.np,
            ],
            [1, 1, 2, 2],
        )
        constraints0 = make_parameter_constraints(module, dtype, state, state.ndx, 1)
        constraints1 = make_parameter_constraints(module, dtype, state, state.ndx, 2)
        tau = [
            np.array([0.1], dtype=dtype),
            np.array([0.2], dtype=dtype),
            np.array([0.3], dtype=dtype),
        ]
        x0 = np.linspace(-0.2, 0.2, 4, dtype=dtype)
        problem = module.ObservationProblem(
            x0=x0,
            tauMeas=tau,
            modelPhases=[[observer0a, observer0b], [observer1]],
            terminalModel=terminal,
            paramsModel=[params0, params1],
            paramsConstraints=[constraints0, constraints1],
        )

        self.assertIsInstance(problem, module.ProblemAbstract)
        self.assertNotIsInstance(problem, module.ShootingProblem)
        for name in (
            "circularAppend",
            "updateNode",
            "updateModel",
            "set_runningModels",
            "set_terminalModel",
        ):
            self.assertFalse(hasattr(problem, name))
        with self.assertRaises(AttributeError):
            problem.runningModels = [observer0a, observer0b, observer1]
        with self.assertRaises(AttributeError):
            problem.terminalModel = terminal
        self.assertEqual((problem.T, problem.n_phases), (3, 2))
        self.assertEqual(list(problem.phase_idxs), [0, 2])
        self.assertEqual(list(problem.phase_edxs), [2, 3])
        self.assertEqual(len(problem.runningPhaseModels[0]), 2)
        self.assertEqual(len(problem.runningPhaseDatas[1]), 1)
        self.assertIs(problem.runningPhaseModels[1][0], observer1)
        problem.runningPhaseDatas[1][0].cost = 1.0
        self.assertEqual(problem.runningDatas[2].cost, 1.0)
        self.assertTrue(np.array_equal(observer0a.tau_meas, tau[0]))
        self.assertTrue(np.array_equal(observer0b.tau_meas, tau[1]))
        self.assertTrue(np.array_equal(observer1.tau_meas, tau[2]))

        p0 = np.array([0.25], dtype=dtype)
        p1 = np.array([-0.2, 0.4], dtype=dtype)
        item0_updates = item0.update_calls
        item1_updates = item1.update_calls
        problem.update_p(p0, 0)
        problem.update_p(p1, 1)
        self.assertEqual(
            (item0.update_calls, item1.update_calls),
            (item0_updates + 1, item1_updates + 1),
        )
        self.assertTrue(np.array_equal(problem.paramsData[0].params.p, p0))
        self.assertTrue(np.array_equal(problem.paramsData[1].params.p, p1))
        ws = [
            np.linspace(0.1, 0.4, 4, dtype=dtype),
            np.linspace(-0.4, -0.1, 4, dtype=dtype),
            np.linspace(0.5, 0.8, 4, dtype=dtype),
        ]
        xs = problem.rollout(ws)
        self.assertFalse(np.allclose(xs[1], x0))
        cost = problem.calc(xs, ws)
        self.assertGreater(cost, 0)
        self.assertEqual(problem.calcDiff(xs, ws), cost)
        self.assertTrue(np.allclose(problem.runningDatas[0].Fu, 0))
        self.assertTrue(np.allclose(problem.runningDatas[0].Fp, 1))
        self.assertTrue(np.allclose(problem.runningDatas[2].Fp[:, 1], 2))
        phase1_data = problem.paramsConstraintsData[1]
        self.assertTrue(np.allclose(phase1_data.h[: state.ndx], ws[2]))
        self.assertTrue(np.allclose(phase1_data.h[state.ndx :], p1))
        self.assertTrue(np.array_equal(phase1_data.Hu[: state.ndx], np.eye(4)))
        self.assertTrue(np.array_equal(phase1_data.Hp[state.ndx :], np.eye(2)))

        problem.update_tau(1, np.array([0.7], dtype=dtype))
        self.assertTrue(np.allclose(observer0b.tau_meas, [0.7]))
        updated_tau = [np.array([-0.1], dtype=dtype) for _ in range(3)]
        problem.update_us(updated_tau)
        self.assertTrue(np.allclose(observer0a.tau_meas, [-0.1]))
        self.assertTrue(np.allclose(observer1.tau_meas, [-0.1]))

        shallow = copy.copy(problem)
        deep = copy.deepcopy(problem)
        shallow.runningDatas[0].cost = 7.0
        self.assertEqual(problem.runningDatas[0].cost, 7.0)
        deep.update_p(p1 / dtype(2), 1)
        self.assertTrue(np.array_equal(problem.paramsData[1].params.p, p1 / dtype(2)))
        problem.update_p(p1, 1)
        self.assertEqual(shallow.calc(xs, ws), problem.calc(xs, ws))

        with self.assertRaises(crocoddyl.Exception):
            problem.update_tau(3, np.zeros(1, dtype=dtype))
        with self.assertRaises(crocoddyl.Exception):
            problem.update_tau(0, np.zeros(2, dtype=dtype))
        with self.assertRaises(crocoddyl.Exception):
            problem.update_us(updated_tau[:-1])
        with self.assertRaises(crocoddyl.Exception):
            problem.calc(xs.tolist()[:-1], ws)
        with self.assertRaises(crocoddyl.Exception):
            module.ObservationProblem(x0, [], [observer0a], terminal, params0)
        with self.assertRaises(crocoddyl.Exception):
            module.ObservationProblem(x0, tau[:1], [None], terminal, params0)
        with self.assertRaises(crocoddyl.Exception):
            module.ObservationProblem(x0, tau[:1], [observer1], observer1, params0)

    def test_float64(self):
        self.check_parametrized_shooting(crocoddyl, np.float64)
        self.check_ordinary_shooting_mutations(crocoddyl, np.float64)
        self.check_observation_problem(crocoddyl, np.float64)

    def test_float32(self):
        self.check_parametrized_shooting(crocoddyl_float32, np.float32)
        self.check_ordinary_shooting_mutations(crocoddyl_float32, np.float32)
        self.check_observation_problem(crocoddyl_float32, np.float32)

    def test_lifetime_subprocesses(self):
        script = r"""
import copy
import gc
import numpy as np
import crocoddyl
import crocoddyl.float32 as crocoddyl_float32

for module, dtype in ((crocoddyl, np.float64), (crocoddyl_float32, np.float32)):
    model = module.ActionModelLQR(4, 2, 1, 0, 0)
    params = module.ParameterManager(model.state)
    params.addParam("lqr", module.LQRParams(model.state, 1))
    problem = module.ShootingProblem(
        np.zeros(4, dtype=dtype), [model], model, params
    )
    pdata = problem.paramsData[0]
    rdata = problem.runningDatas[0]
    retained_model = problem.runningModels[0]
    copied = copy.copy(problem)
    del problem
    del params
    del model
    gc.collect()
    copied.update_p(np.array([0.3], dtype=dtype))
    retained_model.calc(
        rdata, np.zeros(4, dtype=dtype), np.zeros(2, dtype=dtype)
    )
    assert np.array_equal(pdata.params.p, np.array([0.3], dtype=dtype))
"""
        environment = os.environ.copy()
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
