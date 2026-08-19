###############################################################################
# BSD 3-Clause License
#
# Copyright (C) 2026-2026, Heriot-Watt University
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.
###############################################################################

import pickle
import tempfile
import unittest

import numpy as np

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


class SolverProbe:
    def __init__(self, dtype):
        self.xs = [np.zeros(2, dtype=dtype), np.ones(2, dtype=dtype)]
        self.us = [np.zeros(1, dtype=dtype)]
        self.fs = [np.zeros(2, dtype=dtype)]
        self.p = [
            np.array([0.2, -0.3], dtype=dtype),
            np.array([0.4], dtype=dtype),
        ]
        self.Vpp_phase = [
            np.eye(2, dtype=dtype),
            2 * np.eye(1, dtype=dtype),
        ]
        self.iter = 0
        self.cost = 1.5
        self.preg = 1e-6
        self.dreg = 2e-6
        self.stepLength = 0.5
        self.ffeas = 3e-4
        self.hfeas = 4e-4

    def stoppingCriteria(self):
        return 5e-5

    def expectedImprovement(self):
        return np.array([0.0, -0.25])


class IdentityParametrization:
    @staticmethod
    def fromParametrization(data, psi, p):
        psi[:] = p

    @staticmethod
    def updateParametrizationDerivative(data, dpsi_dp, p, psi):
        dpsi_dp[:, :] = np.eye(10)


class PlotSolverProbe:
    def __init__(self, log, p_log, precision_log):
        self.xs = [np.zeros(2), np.ones(2)]
        self.us = [np.zeros(1), np.zeros(0)]
        self.p = [p_log[-1][0], p_log[-1][1]]
        self.Vpp_phase = [precision_log[-1][0], precision_log[-1][1]]
        self._callbacks = [object(), log]

    def getCallbacks(self):
        return self._callbacks


class StateProbe:
    @staticmethod
    def diff(x0, x1):
        return x1 - x0


class CallbackAndPlottingTest(unittest.TestCase):
    def test_parameter_history_owns_every_phase(self):
        for dtype in (np.float64, np.float32):
            with self.subTest(dtype=dtype):
                solver = SolverProbe(dtype)
                logger = crocoddyl.CallbackLogger()
                logger(solver)
                first_p = [p.copy() for p in logger.p[0]]
                first_Vpp = [Vpp.copy() for Vpp in logger.Vpp_phase[0]]

                solver.p[0][0] = dtype(9)
                solver.Vpp_phase[1][0, 0] = dtype(8)
                solver.iter = 1
                logger(solver)

                self.assertEqual(len(logger.p), 2)
                self.assertEqual(len(logger.p[0]), 2)
                self.assertEqual(len(logger.Vpp_phase[0]), 2)
                for actual, expected in zip(logger.p[0], first_p):
                    np.testing.assert_array_equal(actual, expected)
                for actual, expected in zip(logger.Vpp_phase[0], first_Vpp):
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual(logger.p[0][0].dtype, dtype)
                self.assertEqual(logger.Vpp_phase[0][0].dtype, dtype)

                del solver.p
                del solver.Vpp_phase
                logger(solver)
                self.assertEqual(len(logger.p), 2)
                self.assertEqual(len(logger.Vpp_phase), 2)

    def test_parameter_history_from_scalar_solvers(self):
        for module, dtype in (
            (crocoddyl, np.float64),
            (crocoddyl_float32, np.float32),
        ):
            with self.subTest(module=module.__name__):
                running = module.ActionModelLQR(1, 1, 1, 0, 0, True)
                terminal = module.ActionModelLQR(1, 0, 1, 0, 0, True)
                manager = module.ParameterManager(running.state)
                manager.addParam("lqr", module.LQRParams(running.state, 1))
                params_model = module.ParameterPhaseModel(manager)
                problem = module.ShootingProblem(
                    np.zeros(1, dtype=dtype), [running], terminal, params_model
                )
                us = [np.zeros(1, dtype=dtype)]
                solver = module.SolverFDDP(problem)
                logger = module.CallbackLogger()
                solver.setCallbacks([logger])
                solver.solve(
                    problem.rollout(us),
                    us,
                    [np.array([0.2], dtype=dtype)],
                    1,
                    True,
                )

                self.assertEqual(len(logger.p), 1)
                self.assertEqual(len(logger.Vpp_phase), 1)
                self.assertEqual(logger.p[0][0].dtype, dtype)
                self.assertEqual(logger.Vpp_phase[0][0].dtype, dtype)

    def make_covariance_case(self, dtype):
        inertial0 = np.array(
            [2.0, 0.2, -0.4, 0.6, 1.0, 0.1, 1.1, 0.2, 0.3, 1.2],
            dtype=dtype,
        )
        inertial1 = inertial0 + dtype(0.1)
        p_log = [
            [np.concatenate((np.array([0.5, -0.5]), inertial0)), np.array([0.2])],
            [np.concatenate((np.array([0.4, -0.4]), inertial1)), np.array([0.3])],
        ]
        precision_log = [
            [4 * np.eye(12, dtype=dtype), 3 * np.eye(1, dtype=dtype)],
            [5 * np.eye(12, dtype=dtype), 6 * np.eye(1, dtype=dtype)],
        ]
        log = crocoddyl.CallbackLogger()
        log.p = [[p.copy() for p in entry] for entry in p_log]
        log.Vpp_phase = [
            [precision.copy() for precision in entry] for entry in precision_log
        ]
        log.pregs = [0.0, 0.0]
        log.xs = [np.zeros(2), np.ones(2)]
        log.costs = [2.0, 1.0]
        log.dregs = [0.0, 0.0]
        log.grads = [1.0, 0.5]
        log.stops = [0.2, 0.1]
        log.steps = [0.5, 1.0]
        return PlotSolverProbe(log, p_log, precision_log), inertial0, inertial1

    def test_covariance_uses_complete_iteration_history(self):
        for dtype in (np.float64, np.float32):
            with self.subTest(dtype=dtype):
                solver, inertial0, inertial1 = self.make_covariance_case(dtype)
                covariance = crocoddyl.computeInertialCovariances(
                    solver,
                    IdentityParametrization(),
                    None,
                    1,
                    parameter_slice=slice(2, 12),
                )
                self.assertEqual(covariance["parameters_re"].shape, (10, 2))
                self.assertEqual(covariance["parameters_std"].shape, (10, 2))
                np.testing.assert_allclose(
                    covariance["parameters_re"][0],
                    [inertial0[0], inertial1[0]],
                )
                np.testing.assert_allclose(
                    covariance["parameters_re"][1],
                    [inertial0[1] / inertial0[0], inertial1[1] / inertial1[0]],
                )
                self.assertTrue(np.all(np.isfinite(covariance["parameters_std"])))
                self.assertEqual(len(covariance["diagnostics"]), 2)

                initial = solver.p[0].copy()
                covariance = crocoddyl.computeInertialCovariances(
                    solver,
                    IdentityParametrization(),
                    None,
                    1,
                    initial_p=initial,
                    parameter_slice=slice(2, 12),
                )
                self.assertEqual(covariance["x0"], -1)
                self.assertEqual(covariance["parameters_re"].shape, (10, 3))
                self.assertTrue(np.all(np.isnan(covariance["parameters_std"][:, 0])))

    def test_headless_estimation_and_friction_plots(self):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError as error:
            self.skipTest(str(error))

        solver, inertial0, _ = self.make_covariance_case(np.float64)
        covariance = crocoddyl.computeInertialCovariances(
            solver,
            IdentityParametrization(),
            None,
            1,
            parameter_slice=slice(2, 12),
        )
        control_solver = type("ControlSolver", (), {"xs": solver.xs})()
        plt.close("all")
        crocoddyl.plotInertialEstimationWithCovariance(
            solver,
            control_solver,
            StateProbe(),
            IdentityParametrization(),
            None,
            inertial0,
            1,
            covariance,
            show=False,
        )
        self.assertTrue(set(range(1, 7)).issubset(plt.get_fignums()))

        crocoddyl.plotFrictionParam(
            np.log([[0.4, 2.0], [0.5, 2.5]]),
            crocoddyl.JointFrictionType.COULOMB,
            nominal=np.log([[0.3, 1.5], [0.6, 3.0]]),
            figIndex=7,
            joint_name=["joint 1", "joint 2"],
            parametrized=True,
            show=False,
        )
        self.assertIn(7, plt.get_fignums())
        for axis in plt.figure(7).axes:
            for line in axis.lines:
                self.assertTrue(np.all(np.isfinite(line.get_ydata())))

        friction_cases = (
            (crocoddyl.JointFrictionType.COULOMB, np.log([0.4, 2.0]), None),
            (crocoddyl.JointFrictionType.VISCOUS, np.log([0.2]), None),
            (crocoddyl.JointFrictionType.STRIBECK, np.array([1.0, 2.0]), None),
            (
                crocoddyl.JointFrictionType.COULOMB_VISCOUS,
                np.log([0.4, 2.0, 0.2]),
                None,
            ),
            (
                crocoddyl.JointFrictionType.COULOMB_STRIBECK,
                np.array([1.0, 2.0, 0.4, 2.0]),
                None,
            ),
            (
                crocoddyl.JointFrictionType.VISCOUS_STRIBECK,
                np.array([1.0, 2.0, np.log(0.2)]),
                None,
            ),
            (
                crocoddyl.JointFrictionType.FULL,
                np.array([0.5, 1.0, 2.0, 0.4, 2.0, np.log(0.2)]),
                None,
            ),
            (
                crocoddyl.JointFrictionType.COULOMB_FIXED_SMOOTHING,
                np.log([0.4]),
                np.log([2.0]),
            ),
            (
                crocoddyl.JointFrictionType.STRIBECK_FIXED_SMOOTHING,
                np.zeros(0),
                np.array([1.0, 2.0]),
            ),
            (
                crocoddyl.JointFrictionType.COULOMB_VISCOUS_FIXED_SMOOTHING,
                np.log([0.4, 0.2]),
                np.log([2.0]),
            ),
            (
                crocoddyl.JointFrictionType.COULOMB_STRIBECK_FIXED_SMOOTHING,
                np.array([0.4]),
                np.array([1.0, 2.0, 2.0]),
            ),
            (
                crocoddyl.JointFrictionType.VISCOUS_STRIBECK_FIXED_SMOOTHING,
                np.log([0.2]),
                np.array([1.0, 2.0]),
            ),
            (
                crocoddyl.JointFrictionType.FULL_FIXED_SMOOTHING,
                np.array([0.5, 0.4, np.log(0.2)]),
                np.array([1.0, 2.0, 2.0]),
            ),
        )
        for index, (friction_type, parameters, fixed) in enumerate(friction_cases):
            figure_index = 8 + index
            crocoddyl.plotFrictionParam(
                parameters,
                friction_type,
                nominal=parameters,
                figIndex=figure_index,
                fixed_smoothing=fixed,
                parametrized=True,
                show=False,
            )
            for line in plt.figure(figure_index).axes[0].lines:
                self.assertTrue(np.all(np.isfinite(line.get_ydata())))
        self.assertFalse(hasattr(crocoddyl, "plotFritionParam"))
        with self.assertRaisesRegex(ValueError, "fixed_smoothing"):
            crocoddyl.plotFrictionParam(
                np.zeros((1, 1)),
                crocoddyl.JointFrictionType.COULOMB_FIXED_SMOOTHING,
                fixed_smoothing=np.zeros((1, 2)),
                show=False,
            )
        plt.close("all")

    def test_saved_log_contains_parameter_history(self):
        solver = SolverProbe(np.float64)
        logger = crocoddyl.CallbackLogger()
        logger(solver)
        with tempfile.NamedTemporaryFile() as output:
            crocoddyl.saveLogfile(output.name, logger)
            output.seek(0)
            data = pickle.load(output)
        self.assertEqual(len(data["p"][0]), 2)
        self.assertEqual(len(data["Vpp_phase"][0]), 2)


if __name__ == "__main__":
    unittest.main()
