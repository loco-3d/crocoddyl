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


class ProblemAbstractTest(unittest.TestCase):
    def check_problem_interface(self, module, dtype):
        model = module.ActionModelLQR(4, 2)
        x0 = np.linspace(-0.2, 0.4, 4, dtype=dtype)
        problem = module.ShootingProblem(x0, [model, model], model)
        problem.nthreads = 1

        self.assertIsInstance(problem, module.ProblemAbstract)
        with self.assertRaises(RuntimeError):
            module.ProblemAbstract()

        xs = [np.zeros(4, dtype=dtype) for _ in range(3)]
        us = [np.zeros(2, dtype=dtype) for _ in range(2)]
        self.assertEqual(problem.calc(xs, us), 0.0)
        self.assertEqual(problem.calcDiff(xs, us), 0.0)
        rollout = problem.rollout(us)
        self.assertEqual(len(rollout), 3)
        self.assertTrue(np.allclose(rollout[0], x0))

        self.assertEqual(problem.T, 2)
        self.assertTrue(np.array_equal(problem.x0, x0))
        self.assertEqual((problem.nx, problem.ndx, problem.nthreads), (4, 4, 1))
        self.assertEqual(len(problem.runningModels), 2)
        self.assertEqual(len(problem.runningDatas), 2)
        self.assertIsInstance(problem.runningModels[0], module.ActionModelAbstract)
        self.assertIsInstance(problem.terminalModel, module.ActionModelAbstract)
        self.assertIsInstance(problem.runningDatas[0], module.ActionDataAbstract)
        self.assertIsInstance(problem.terminalData, module.ActionDataAbstract)

        self.assertFalse(problem.is_updated)
        problem.is_updated = True
        self.assertTrue(problem.is_updated)
        self.assertFalse(problem.is_updated)

        self.assertEqual(problem.n_phases, 0)
        self.assertEqual(len(problem.phase_idxs), 0)
        self.assertEqual(len(problem.phase_edxs), 0)
        self.assertEqual(len(problem.paramsConstraints), 0)
        self.assertEqual(len(problem.paramsConstraintsData), 0)
        self.assertFalse(problem.hasParamsConstraints)
        with self.assertRaises(crocoddyl.Exception):
            problem.update_p(np.zeros(1, dtype=dtype), 0)

        shallow = copy.copy(problem)
        deep = copy.deepcopy(problem)
        self.assertIsInstance(shallow, module.ProblemAbstract)
        self.assertIsInstance(deep, module.ProblemAbstract)
        self.assertEqual((shallow.T, deep.T), (problem.T, problem.T))

    def test_float64(self):
        self.check_problem_interface(crocoddyl, np.float64)

    def test_float32(self):
        self.check_problem_interface(crocoddyl_float32, np.float32)


if __name__ == "__main__":
    unittest.main()
