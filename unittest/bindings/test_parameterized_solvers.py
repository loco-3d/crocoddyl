###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026-2026, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################

import unittest

import numpy as np
from test_parameterized_problems import make_dynamics_params, make_observer

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


class ParameterizedSolversTest(unittest.TestCase):
    def make_problem(self, module, dtype, mixed_constraints=False):
        phase0 = module.ActionModelLQR(4, 2, 1, 0, 1)
        phase1 = module.ActionModelLQR(4, 1, 2, 0, 1)
        terminal = module.ActionModelLQR(4, 0, 2, 0, 0)
        H0 = np.zeros((1, 7), dtype=dtype)
        H0[0, 4] = 1
        H0[0, 6] = 0.4
        phase0.H = H0
        H1 = np.zeros((1, 7), dtype=dtype)
        H1[0, 4] = 1
        H1[0, 5:] = (-0.25, 0.3)
        phase1.H = H1
        params0 = module.ParameterManager(phase0.state)
        params1 = module.ParameterManager(phase1.state)
        params0.addParam("lqr", module.LQRParams(phase0.state, 1))
        params1.addParam("lqr", module.LQRParams(phase1.state, 2))
        params0.addParam("inactive", module.LQRParams(phase0.state, 2), active=False)
        params1.addParam("inactive", module.LQRParams(phase1.state, 1), active=False)
        args = [
            np.zeros(4, dtype=dtype),
            [[phase0, phase0], [phase1, phase1]],
            terminal,
            [params0, params1],
        ]
        if mixed_constraints:
            constraints1 = module.ConstraintModelManager(phase1.state, 1, 2)
            constraints1.addConstraint(
                "parameters",
                module.ConstraintModelResidual(
                    phase1.state,
                    module.ResidualModelParameters(
                        phase1.state, np.zeros(2, dtype=dtype), 1
                    ),
                ),
            )
            args.append([None, constraints1])
        problem = module.ParametrizedShootingProblem(*args)
        us = [
            np.full(2, 0.1, dtype=dtype),
            np.full(2, -0.05, dtype=dtype),
            np.full(1, 0.2, dtype=dtype),
            np.full(1, -0.1, dtype=dtype),
        ]
        return problem, problem.rollout(us), us

    def make_quadratic_problem(self, module, dtype):
        running = module.ActionModelLQR(1, 1, 1, 0, 0, True)
        running.A = np.eye(1, dtype=dtype)
        running.B = np.zeros((1, 1), dtype=dtype)
        running.P = np.zeros((1, 1), dtype=dtype)
        running.Q = np.zeros((1, 1), dtype=dtype)
        running.R = np.eye(1, dtype=dtype)
        running.N = np.zeros((1, 1), dtype=dtype)
        running.W = np.zeros((1, 1), dtype=dtype)
        running.Y = np.zeros((1, 1), dtype=dtype)
        running.V = np.zeros((1, 1), dtype=dtype)
        running.f = np.zeros(1, dtype=dtype)
        running.q = np.zeros(1, dtype=dtype)
        running.r = np.zeros(1, dtype=dtype)
        running.m = np.zeros(1, dtype=dtype)

        terminal = module.ActionModelLQR(1, 0, 1, 0, 0, True)
        terminal.A = np.eye(1, dtype=dtype)
        terminal.P = np.zeros((1, 1), dtype=dtype)
        terminal.Q = np.zeros((1, 1), dtype=dtype)
        terminal.W = dtype(2) * np.eye(1, dtype=dtype)
        terminal.Y = np.zeros((1, 1), dtype=dtype)
        terminal.f = np.zeros(1, dtype=dtype)
        terminal.q = np.zeros(1, dtype=dtype)
        terminal.m = np.array([-0.7], dtype=dtype)

        params = module.ParameterManager(running.state)
        params.addParam("lqr", module.LQRParams(running.state, 1))
        problem = module.ParametrizedShootingProblem(
            np.zeros(1, dtype=dtype), [running, running], terminal, params
        )
        us = [np.zeros(1, dtype=dtype), np.zeros(1, dtype=dtype)]
        return problem, problem.rollout(us), us

    def check_parameterized_solvers(self, module, dtype):
        problem, xs, us = self.make_problem(module, dtype)
        self.assertEqual(problem.params[0].np, 1)
        self.assertEqual(problem.params[1].np, 2)
        self.assertIn("lqr", problem.params[0].active_set)
        self.assertIn("inactive", problem.params[0].inactive_set)
        self.assertIn("lqr", problem.params[1].active_set)
        self.assertIn("inactive", problem.params[1].inactive_set)
        cast_dtype = (
            crocoddyl.DType.Float32 if module is crocoddyl else crocoddyl.DType.Float64
        )
        init_p = [
            np.array([0.2], dtype=dtype),
            np.array([-0.3, 0.4], dtype=dtype),
        ]
        for solver_type in (module.SolverFDDP, module.SolverIntro):
            solver = solver_type(problem)
            self.assertFalse(solver.solve(xs, us, init_p, 0, True))
            for actual, expected in zip(solver.p, init_p):
                self.assertTrue(np.array_equal(actual, expected))
            for actual, expected in zip(solver.p_try, init_p):
                self.assertTrue(np.array_equal(actual, expected))
            solver.computeDirection(True)
            self.assertEqual([value.shape for value in solver.p], [(1,), (2,)])
            self.assertEqual(
                [value.shape for value in solver.Vp],
                [(1,), (1,), (2,), (2,), (2,)],
            )
            self.assertEqual(
                [value.shape for value in solver.Qp],
                [(1,), (1,), (2,), (2,)],
            )
            # EigenPy exposes one-row/one-column dynamic matrices as vectors;
            # C++ coverage checks both matrix dimensions.
            self.assertEqual([value.size for value in solver.Qpu], [2, 2, 2, 2])
            self.assertEqual([value.size for value in solver.P], [2, 2, 2, 2])
            for collection in (
                solver.dp,
                solver.kp,
                solver.Kp,
                solver.Vp,
                solver.Vpp,
                solver.Vpx,
                solver.Qp,
                solver.Qpp,
                solver.Qpx,
                solver.Qpu,
                solver.P,
            ):
                self.assertTrue(all(np.all(np.isfinite(value)) for value in collection))
            steplength = dtype(0.25)
            expected_p_try = [p + steplength * dp for p, dp in zip(solver.p, solver.dp)]
            solver.computeCandidate(float(steplength))
            self.assertEqual([value.shape for value in solver.p_try], [(1,), (2,)])
            for actual, expected in zip(solver.p_try, expected_p_try):
                self.assertTrue(np.allclose(actual, expected))
            problem_before = solver.problem
            p_before = [value.copy() for value in solver.p]
            manager_p_before = [data.params.p.copy() for data in problem.params_data]
            with self.assertRaisesRegex(Exception, "cannot be cast"):
                solver.cast(cast_dtype)
            self.assertIs(solver.problem, problem_before)
            self.assertIs(solver.problem, problem)
            for actual, expected in zip(solver.p, p_before):
                self.assertTrue(np.array_equal(actual, expected))
            for data, expected in zip(problem.params_data, manager_p_before):
                self.assertTrue(np.array_equal(data.params.p, expected))

            with self.assertRaises(Exception):
                solver.solve(xs, us, init_p[:1], 0, True)
            wrong = [init_p[0], init_p[1][:1]]
            with self.assertRaises(Exception):
                solver.solve(xs, us, wrong, 0, True)

            class Callback(module.CallbackAbstract):
                def __init__(self):
                    super().__init__()
                    self.calls = 0

                def __call__(self, current_solver):
                    self.calls += 1
                    self.assert_solver = current_solver

            callback = Callback()
            solver.setCallbacks([callback])
            solver.solve(xs, us, init_p, 1, True, 1e-6)
            self.assertGreaterEqual(callback.calls, 1)
            self.assertIsInstance(callback.assert_solver, solver_type)
            self.assertEqual(callback.assert_solver.problem.T, problem.T)
            self.assertEqual(
                [value.shape for value in callback.assert_solver.p], [(1,), (2,)]
            )

        for eq_solver in (
            crocoddyl.EqualitySolverType.LuNull,
            crocoddyl.EqualitySolverType.QrNull,
            crocoddyl.EqualitySolverType.Schur,
        ):
            solver = module.SolverIntro(
                problem, crocoddyl.DynamicsSolverType.FeasShoot, eq_solver
            )
            solver.solve(xs, us, init_p, 0, True)
            solver.computeDirection(True)
            for model, data, gain in zip(
                problem.runningModels, problem.runningDatas, solver.P
            ):
                Hu = np.asarray(data.Hu).reshape(data.h.size, model.nu)
                Hp = np.asarray(data.Hp).reshape(data.h.size, model.np)
                P = np.asarray(gain).reshape(model.nu, model.np)
                self.assertTrue(np.allclose(Hu @ P, Hp))
            if eq_solver != crocoddyl.EqualitySolverType.Schur:
                for model, rank, Qz, Qzz, Qxz, Quz, Qzc in zip(
                    problem.runningModels,
                    solver.Hu_rank,
                    solver.Qz,
                    solver.Qzz,
                    solver.Qxz,
                    solver.Quz,
                    solver.Qzc,
                ):
                    nz = model.nu - rank
                    self.assertEqual(Qz.shape, (nz,))
                    self.assertEqual(Qzz.shape, (nz, nz))
                    self.assertEqual(np.asarray(Qxz).size, model.state.ndx * nz)
                    self.assertEqual(np.asarray(Quz).size, model.nu * nz)
                    self.assertEqual(
                        np.asarray(Qzc).size, nz * problem.terminalModel.nh_T
                    )

        standard_model = module.ActionModelLQR(4, 2)
        standard_terminal = module.ActionModelLQR(4, 0)
        standard_problem = module.ShootingProblem(
            np.zeros(4, dtype=dtype),
            [standard_model, standard_model],
            standard_terminal,
        )
        for solver_type in (module.SolverFDDP, module.SolverIntro):
            standard = solver_type(standard_problem)
            with self.assertRaises(TypeError):
                standard.solve([], [], 1)
            self.assertFalse(standard.solve([], [], maxiter=0))
            casted = standard.cast(cast_dtype)
            self.assertEqual(casted.problem.T, standard_problem.T)

        for solver_type in (module.SolverFDDP, module.SolverIntro):
            with self.assertRaisesRegex(Exception, "problem is null"):
                solver_type(None)

    def check_quadratic_parameter_optimization(self, module, dtype):
        tol = 2e-4 if dtype is np.float32 else 1e-9
        initial = np.array([-0.4], dtype=dtype)
        optimum = np.array([0.35], dtype=dtype)
        for solver_type in (module.SolverFDDP, module.SolverIntro):
            problem, xs, us = self.make_quadratic_problem(module, dtype)
            trial = solver_type(problem)
            self.assertFalse(trial.solve(xs, us, [initial], 0, True))
            trial.computeDirection(True)
            trial.expectedImprovement()
            trial.tryStep(0.0)
            self.assertFalse(trial.checkAcceptance())
            trial.tryStep(0.5)
            self.assertTrue(trial.checkAcceptance())
            self.assertAlmostEqual(float(trial.dV), float(trial.dVexp), delta=tol)

            solver = solver_type(problem)
            self.assertTrue(solver.solve(xs, us, [initial], 10, True))
            self.assertGreater(solver.iter, 0)
            self.assertTrue(np.allclose(solver.p[0], optimum, atol=tol, rtol=tol))
            self.assertTrue(
                np.allclose(
                    problem.params_data[0].params.p,
                    optimum,
                    atol=tol,
                    rtol=tol,
                )
            )
            recomputed_cost = problem.calc(solver.xs, solver.us)
            self.assertAlmostEqual(
                float(recomputed_cost), float(solver.cost), delta=tol
            )
            problem.calcDiff(solver.xs, solver.us)
            self.assertTrue(np.allclose(problem.terminalData.Lp, 0, atol=tol, rtol=tol))
            self.assertAlmostEqual(float(recomputed_cost), -0.1225, delta=tol)

    def check_mixed_parameter_constraint_restoration(self, module, dtype):
        tol = 2e-5 if dtype is np.float32 else 1e-12
        problem, xs, us = self.make_problem(module, dtype, mixed_constraints=True)
        initial = [
            np.array([0.4], dtype=dtype),
            np.array([-0.5, 0.6], dtype=dtype),
        ]
        solver = module.SolverFDDP(problem)
        self.assertFalse(solver.solve(xs, us, initial, 0, True))
        solver.computeDirection(True)
        self.assertIsNone(problem.parameter_constraints[0])
        self.assertIsNone(problem.parameter_constraints_data[0])
        self.assertIsNotNone(problem.parameter_constraints[1])
        self.assertIsNotNone(problem.parameter_constraints_data[1])

        accepted = [value.copy() for value in solver.p]
        candidate0 = accepted[0] + dtype(0.25)
        candidate1 = accepted[1] + dtype(0.35)
        problem.update_p(candidate0, 0)
        problem.update_p(candidate1, 1)
        constraints = problem.parameter_constraints[1]
        constraints_data = problem.parameter_constraints_data[1]
        constraints.calc(
            constraints_data,
            np.zeros(4, dtype=dtype),
            np.zeros(1, dtype=dtype),
        )
        candidate_h = constraints_data.h.copy()
        solver.computeDirection(True)
        self.assertTrue(
            np.allclose(problem.params_data[0].params.p, accepted[0], atol=tol)
        )
        self.assertTrue(
            np.allclose(problem.params_data[1].params.p, accepted[1], atol=tol)
        )
        self.assertFalse(np.allclose(constraints_data.h, candidate_h, atol=tol))
        self.assertTrue(np.allclose(constraints_data.h, accepted[1], atol=tol))

    def check_observation_problem(self, module, dtype):
        state = module.StateVector(4)
        params, item = make_dynamics_params(module, dtype, state, 1)
        observer0, _ = make_observer(module, dtype, state, 1, item)
        observer1, _ = make_observer(module, dtype, state, 1, item)
        terminal, _ = make_observer(module, dtype, state, 1, item)
        tau = [np.array([0.1], dtype=dtype), np.array([0.2], dtype=dtype)]
        problem = module.ObservationProblem(
            np.zeros(4, dtype=dtype),
            tau,
            [observer0, observer1],
            terminal,
            params,
        )
        process_noise = [
            np.linspace(0.1, 0.4, 4, dtype=dtype),
            np.linspace(-0.4, -0.1, 4, dtype=dtype),
        ]
        xs = problem.rollout(process_noise)
        solver = module.SolverFDDP(problem)
        init_p = [np.array([0.35], dtype=dtype)]
        solver.solve(xs, process_noise, init_p, 0, True)
        solver.computeDirection(True)
        self.assertEqual(solver.p[0].shape, (1,))
        self.assertTrue(np.all(np.isfinite(solver.dp[0])))

    def test_float64(self):
        self.check_parameterized_solvers(crocoddyl, np.float64)
        self.check_quadratic_parameter_optimization(crocoddyl, np.float64)
        self.check_mixed_parameter_constraint_restoration(crocoddyl, np.float64)
        self.check_observation_problem(crocoddyl, np.float64)

    def test_float32(self):
        self.check_parameterized_solvers(crocoddyl_float32, np.float32)
        self.check_quadratic_parameter_optimization(crocoddyl_float32, np.float32)
        self.check_mixed_parameter_constraint_restoration(crocoddyl_float32, np.float32)
        self.check_observation_problem(crocoddyl_float32, np.float32)


if __name__ == "__main__":
    unittest.main()
