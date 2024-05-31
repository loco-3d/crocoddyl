import sys
import unittest
from random import randint

import example_robot_data
import numpy as np
import pinocchio
from factory import DDPDerived, FDDPDerived

import crocoddyl


class SolverAbstractTestCase(unittest.TestCase):
    MODEL = None
    SOLVER = None
    SOLVER_DER = None

    def setUp(self):
        # Set up the solvers
        self.T = randint(1, 21)
        state = self.MODEL.state
        self.xs = []
        self.us = []
        self.xs.append(state.rand())
        rng = np.random.default_rng()
        for _ in range(self.T):
            self.xs.append(state.rand())
            self.us.append(rng.random(self.MODEL.nu))
        self.PROBLEM = crocoddyl.ShootingProblem(
            self.xs[0], [self.MODEL] * self.T, self.MODEL
        )
        self.PROBLEM_DER = crocoddyl.ShootingProblem(
            self.xs[0], [self.MODEL] * self.T, self.MODEL
        )
        self.solver = self.SOLVER(self.PROBLEM)
        self.solver_der = self.SOLVER_DER(self.PROBLEM_DER)

    def test_number_of_nodes(self):
        # Check the number of nodes
        self.assertEqual(
            self.T, self.solver.problem.T, "Wrong number of nodes in SOLVER"
        )
        self.assertEqual(
            self.T, self.solver_der.problem.T, "Wrong number of nodes in SOLVER_DER"
        )

    def test_solve(self):
        # Run maximum 10 iterations in order to boost test analysis
        self.solver.solve([], [], 10)
        self.solver_der.solve([], [], 10)
        for x1, x2 in zip(self.solver.xs, self.solver_der.xs):
            self.assertTrue(np.allclose(x1, x2, atol=1e-9), "xs doesn't match.")
        for u1, u2 in zip(self.solver.us, self.solver_der.us):
            self.assertTrue(np.allclose(u1, u2, atol=1e-9), "us doesn't match.")
        for k1, k2 in zip(self.solver.k, self.solver_der.k):
            self.assertTrue(np.allclose(k1, k2, atol=1e-9), "k doesn't match.")

    def test_compute_search_direction(self):
        # Compute the direction
        self.solver.setCandidate([], [], False)
        self.solver_der.setCandidate([], [], False)
        self.solver.computeDirection()
        self.solver_der.computeDirection()
        # Check the LQ model of the Hamiltonian
        for qx1, qx2 in zip(self.solver.Qx, self.solver_der.Qx):
            self.assertTrue(np.allclose(qx1, qx2, atol=1e-9), "Qx doesn't match.")
        for qu1, qu2 in zip(self.solver.Qu, self.solver_der.Qu):
            self.assertTrue(np.allclose(qu1, qu2, atol=1e-9), "Qu doesn't match.")
        for qxx1, qxx2 in zip(self.solver.Qxx, self.solver_der.Qxx):
            self.assertTrue(np.allclose(qxx1, qxx2, atol=1e-9), "Qxx doesn't match.")
        for qxu1, qxu2 in zip(self.solver.Qxu, self.solver_der.Qxu):
            self.assertTrue(np.allclose(qxu1, qxu2, atol=1e-9), "Quu doesn't match.")
        for quu1, quu2 in zip(self.solver.Quu, self.solver_der.Quu):
            self.assertTrue(np.allclose(quu1, quu2, atol=1e-9), "Quu doesn't match.")
        for vx1, vx2 in zip(self.solver.Vx, self.solver_der.Vx):
            self.assertTrue(np.allclose(vx1, vx2, atol=1e-9), "Vx doesn't match.")
        for vxx1, vxx2 in zip(self.solver.Vxx, self.solver_der.Vxx):
            self.assertTrue(np.allclose(vxx1, vxx2, atol=1e-9), "Vxx doesn't match.")

    def test_try_step(self):
        # Try a full step and check the improvement in the cost
        self.solver.setCandidate([], [], False)
        self.solver_der.setCandidate([], [], False)
        self.solver.computeDirection()
        self.solver_der.computeDirection()
        cost = self.solver.tryStep()
        costDer = self.solver_der.tryStep()
        self.assertAlmostEqual(cost, costDer, 9, "Wrong cost value for full step")
        # Try a half step and check the improvement in the cost
        cost = self.solver.tryStep(0.5)
        costDer = self.solver_der.tryStep(0.5)
        self.assertAlmostEqual(cost, costDer, 9, "Wrong cost value for half step")

    def test_stopping_criteria(self):
        # Run 2 iteration in order to boost test analysis
        self.solver.solve([], [], 2)
        self.solver_der.solve([], [], 2)
        # Compute and check the stopping criteria
        stop = self.solver.stoppingCriteria()
        stopDer = self.solver_der.stoppingCriteria()
        self.assertAlmostEqual(stop, stopDer, 9, "Wrong stopping value")

    def test_expected_improvement(self):
        # Run 2 iteration in order to boost test analysis
        self.solver.solve([], [], 2)
        self.solver_der.solve([], [], 2)
        expImp = self.solver.expectedImprovement()
        expImpDer = self.solver_der.expectedImprovement()
        self.assertTrue(
            np.allclose(expImp, expImpDer, atol=1e-9),
            "Expected improvement doesn't match.",
        )


class UnicycleDDPTest(SolverAbstractTestCase):
    MODEL = crocoddyl.ActionModelUnicycle()
    SOLVER = crocoddyl.SolverDDP
    SOLVER_DER = DDPDerived


class UnicycleFDDPTest(SolverAbstractTestCase):
    MODEL = crocoddyl.ActionModelUnicycle()
    SOLVER = crocoddyl.SolverFDDP
    SOLVER_DER = FDDPDerived


class TalosArmDDPTest(SolverAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("talos_arm").model
    STATE = crocoddyl.StateMultibody(ROBOT_MODEL)
    ACTUATION = crocoddyl.ActuationModelFull(STATE)
    COST_SUM = crocoddyl.CostModelSum(STATE)
    COST_SUM.addCost(
        "gripperPose",
        crocoddyl.CostModelResidual(
            STATE,
            crocoddyl.ResidualModelFramePlacement(
                STATE,
                ROBOT_MODEL.getFrameId("gripper_left_joint"),
                pinocchio.SE3.Random(),
            ),
        ),
        1e-5,
    )
    COST_SUM.addCost(
        "xReg",
        crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelState(STATE)),
        1e-7,
    )
    COST_SUM.addCost(
        "uReg",
        crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelControl(STATE)),
        1e-7,
    )
    DIFF_MODEL = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        STATE, ACTUATION, COST_SUM
    )
    MODEL = crocoddyl.IntegratedActionModelEuler(DIFF_MODEL, 1e-3)
    SOLVER = crocoddyl.SolverDDP
    SOLVER_DER = DDPDerived


class TalosArmFDDPTest(SolverAbstractTestCase):
    ROBOT_MODEL = example_robot_data.load("talos_arm").model
    STATE = crocoddyl.StateMultibody(ROBOT_MODEL)
    ACTUATION = crocoddyl.ActuationModelFull(STATE)
    COST_SUM = crocoddyl.CostModelSum(STATE)
    COST_SUM.addCost(
        "gripperPose",
        crocoddyl.CostModelResidual(
            STATE,
            crocoddyl.ResidualModelFramePlacement(
                STATE,
                ROBOT_MODEL.getFrameId("gripper_left_joint"),
                pinocchio.SE3.Random(),
            ),
        ),
        1e-5,
    )
    COST_SUM.addCost(
        "xReg",
        crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelState(STATE)),
        1e-7,
    )
    COST_SUM.addCost(
        "uReg",
        crocoddyl.CostModelResidual(STATE, crocoddyl.ResidualModelControl(STATE)),
        1e-7,
    )
    DIFF_MODEL = crocoddyl.DifferentialActionModelFreeFwdDynamics(
        STATE, ACTUATION, COST_SUM
    )
    MODEL = crocoddyl.IntegratedActionModelEuler(DIFF_MODEL, 1e-3)
    SOLVER = crocoddyl.SolverFDDP
    SOLVER_DER = FDDPDerived


if __name__ == "__main__":
    # test to be run
    test_classes_to_run = [
        UnicycleDDPTest,
        UnicycleFDDPTest,
        TalosArmDDPTest,
        TalosArmFDDPTest,
    ]
    loader = unittest.TestLoader()
    suites_list = []
    for test_class in test_classes_to_run:
        suite = loader.loadTestsFromTestCase(test_class)
        suites_list.append(suite)
    big_suite = unittest.TestSuite(suites_list)
    runner = unittest.TextTestRunner()
    results = runner.run(big_suite)
    sys.exit(not results.wasSuccessful())
