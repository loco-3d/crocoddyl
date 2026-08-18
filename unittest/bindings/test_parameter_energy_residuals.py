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

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


class ParameterEnergyResidualBindingsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state64 = crocoddyl.StateMultibody(pinocchio.buildSampleModelManipulator())
        parametrization = crocoddyl.LogCholeskyParametrization()
        workspace = parametrization.createData()
        seed = np.array([0.2, -0.1, 0.15, -0.2, 0.1, -0.25, 0.3, 0.05, -0.08, 0.12])
        physical = np.empty(10)
        parametrization.fromParametrization(workspace, physical, seed)
        cls.state64.pinocchio.inertias[1] = pinocchio.Inertia.FromDynamicParameters(
            physical
        )
        cls.body_names = [
            cls.state64.pinocchio.names[1],
            cls.state64.pinocchio.names[2],
        ]
        cls.unselected_mass = pinocchio.computeTotalMass(cls.state64.pinocchio) - sum(
            cls.state64.pinocchio.inertias[i].mass for i in (1, 2)
        )
        cls.joint_nq = cls.state64.pinocchio.joints[1].nq
        cls.state32 = cls.state64.cast(crocoddyl.DType.Float32)

    def scalar_cases(self):
        return (
            (crocoddyl, np.float64, self.state64, crocoddyl.DType.Float32),
            (
                crocoddyl_float32,
                np.float32,
                self.state32,
                crocoddyl.DType.Float64,
            ),
        )

    @staticmethod
    def make_fixture(module, dtype, state, joint_nq, body_names):
        joint_id = 1
        mu = np.array([np.log(0.3), np.log(4.0)], dtype=dtype)
        friction = module.JointDynamicsModelFriction(
            joint_id,
            joint_nq,
            mu,
            crocoddyl.JointFrictionType.COULOMB,
        )
        actuation = module.ActuationModelMultibody(state, [friction])
        actuation_params = module.ActuationMultibodyParams(actuation)
        parametrization = module.LogCholeskyParametrization()
        inertial_params = module.MultibodyInertialParams(
            state, parametrization, body_names
        )
        inactive_inertial = module.MultibodyInertialParams(
            state, parametrization, body_names
        )
        time = module.IntegratorTime(0.02, True)
        time_params = module.IntegratorTimeoptParams(state, time)
        inactive_time = module.IntegratorTimeoptParams(state, time)

        manager = module.ParameterManager(state)
        manager.addParam("a_inactive_time", inactive_time, False)
        manager.addParam("z_time", time_params)
        manager.addParam("a_actuation", actuation_params)
        manager.addParam("m_inactive_inertial", inactive_inertial, False)
        manager.addParam("z_inertial", inertial_params)
        assert (manager.np_action, manager.np_dynamics, manager.np) == (1, 22, 23)
        manager_data = manager.createData()
        p = np.asarray(manager.zero(), dtype=dtype)
        p[:3] += np.array([0.04, 0.08, -0.03], dtype=dtype)
        p[3:] += dtype(0.01)
        manager.update(manager_data, p)

        implicit = module.ImplicitConstraintModelMultiple(state, actuation.nu)
        dynamics = module.DynamicsModelConstrainedForward(state, actuation, implicit)
        dynamics_data = dynamics.createData(manager_data)
        dynamics.set_params(dynamics_data, manager)
        dynamics.update_p(dynamics_data, p)
        shared = dynamics_data.shared
        x = np.asarray(
            state.integrate(
                state.zero(), np.linspace(-0.08, 0.11, state.ndx, dtype=dtype)
            ),
            dtype=dtype,
        )
        u = np.full(actuation.nu, dtype(0.2), dtype=dtype)
        return {
            "actuation": actuation,
            "manager": manager,
            "manager_data": manager_data,
            "dynamics": dynamics,
            "dynamics_data": dynamics_data,
            "shared": shared,
            "p": p,
            "x": x,
            "u": u,
        }

    def test_parameter_residuals_layout_terminal_copy_and_cast(self):
        for module, dtype, state, cast_dtype in self.scalar_cases():
            with self.subTest(module=module.__name__):
                fixture = self.make_fixture(
                    module, dtype, state, self.joint_nq, self.body_names
                )
                manager = fixture["manager"]
                manager_data = fixture["manager_data"]
                x, u, p = fixture["x"], fixture["u"], fixture["p"]
                np_total = manager.np

                parameters = module.ResidualModelParameters(
                    state, p - dtype(0.2), fixture["actuation"].nu
                )
                parameters_data = parameters.createData(manager_data)
                parameters.calc(parameters_data, x, u)
                parameters.calcDiff(parameters_data, x, u)
                np.testing.assert_allclose(parameters_data.r, dtype(0.2), atol=5e-7)
                np.testing.assert_allclose(
                    parameters_data.Rp, np.eye(np_total, dtype=dtype)
                )
                parameters.calc(parameters_data, x)
                np.testing.assert_allclose(parameters_data.r, dtype(0.2), atol=5e-7)

                actuation_payload = manager_data.dynamics_params["a_actuation"]
                gamma_ref = np.asarray(actuation_payload.gamma) - dtype(0.15)
                actuation = module.ResidualModelActuationParameters(
                    state,
                    gamma_ref,
                    fixture["actuation"].nu,
                    np_total,
                    "a_actuation",
                )
                actuation_data = actuation.createData(manager_data)
                self.assertIsInstance(
                    actuation_data, module.ResidualDataActuationParameters
                )
                self.assertEqual(actuation_data.np_offset, 1)
                actuation.calc(actuation_data, x, u)
                actuation.calcDiff(actuation_data, x, u)
                np.testing.assert_allclose(actuation_data.r, dtype(0.15), atol=1e-5)
                np.testing.assert_allclose(
                    actuation_data.Rp[:, 1:3],
                    actuation_payload.dgamma_dp,
                    atol=2e-5,
                )
                actuation.calc(actuation_data, x)
                np.testing.assert_allclose(actuation_data.r, dtype(0.15), atol=1e-5)

                inertial_payload = manager_data.dynamics_params["z_inertial"]
                psi = np.concatenate(
                    [np.asarray(value, dtype=dtype) for value in inertial_payload.psi]
                )
                inertial = module.ResidualModelInertialParameters(
                    state,
                    psi - dtype(0.1),
                    fixture["actuation"].nu,
                    np_total,
                    "z_inertial",
                )
                inertial_data = inertial.createData(manager_data)
                self.assertIsInstance(
                    inertial_data, module.ResidualDataInertialParameters
                )
                self.assertEqual(inertial_data.np_offset, 3)
                inertial.calc(inertial_data, x, u)
                inertial.calcDiff(inertial_data, x, u)
                np.testing.assert_allclose(inertial_data.r, dtype(0.1), atol=2e-5)
                for i in range(2):
                    np.testing.assert_allclose(
                        inertial_data.Rp[
                            10 * i : 10 * (i + 1), 3 + 10 * i : 13 + 10 * i
                        ],
                        inertial_payload.dpsi_dp[i],
                        atol=3e-5,
                    )
                inertial.calc(inertial_data, x)
                np.testing.assert_allclose(inertial_data.r, dtype(0.1), atol=2e-5)

                S_generic = np.diag(np.linspace(0.25, 1.25, np_total, dtype=dtype))
                generic_symmetry = module.ResidualModelSymmetryParameters(
                    state, S_generic, fixture["actuation"].nu, np_total
                )
                plain_params = module.ParamsDataAbstract(np_total, 0)
                plain_params.p = p.copy()
                plain_collector = module.DataCollectorParams(plain_params)
                generic_symmetry_data = generic_symmetry.createData(plain_collector)
                np.testing.assert_allclose(generic_symmetry_data.Rp, S_generic)
                generic_symmetry.calc(generic_symmetry_data, x, u)
                generic_symmetry_data.Rp = dtype(2) * generic_symmetry_data.Rp
                generic_Rp = generic_symmetry_data.Rp.copy()
                generic_symmetry.calcDiff(generic_symmetry_data, x, u)
                np.testing.assert_allclose(
                    generic_symmetry_data.r, S_generic @ p, atol=3e-5
                )
                np.testing.assert_allclose(generic_symmetry_data.Rp, generic_Rp)
                self.assertEqual(generic_symmetry.param_name, "")
                self.assertEqual(generic_symmetry.cast(cast_dtype).np, np_total)
                np.testing.assert_allclose(copy.copy(generic_symmetry).S, S_generic)

                S_actuation = np.diag(np.array([0.4, 0.8], dtype=dtype))
                actuation_symmetry = module.ResidualModelSymmetryParameters(
                    state,
                    S_actuation,
                    fixture["actuation"].nu,
                    np_total,
                    "a_actuation",
                )
                actuation_symmetry_data = actuation_symmetry.createData(manager_data)
                actuation_symmetry.calc(actuation_symmetry_data, x, u)
                actuation_symmetry.calcDiff(actuation_symmetry_data, x, u)
                np.testing.assert_allclose(
                    actuation_symmetry_data.r,
                    S_actuation @ actuation_payload.gamma,
                    atol=3e-5,
                )
                np.testing.assert_allclose(
                    actuation_symmetry_data.Rp[:, 1:3],
                    S_actuation @ actuation_payload.dgamma_dp,
                    atol=3e-5,
                )

                S = np.zeros((20, 20), dtype=dtype)
                S[:10, :10] = np.diag(np.linspace(0.5, 1.4, 10, dtype=dtype))
                S[:10, 10:] = np.eye(10, dtype=dtype) * dtype(0.15)
                S[10:, :10] = np.eye(10, dtype=dtype) * dtype(-0.2)
                S[10:, 10:] = np.diag(np.linspace(1.5, 2.4, 10, dtype=dtype))
                symmetry = module.ResidualModelSymmetryParameters(
                    state, S, fixture["actuation"].nu, np_total, "z_inertial"
                )
                symmetry_data = symmetry.createData(manager_data)
                self.assertEqual(symmetry_data.np_offset, 3)
                symmetry.calc(symmetry_data, x, u)
                symmetry.calcDiff(symmetry_data, x, u)
                expected_symmetry = S[:, :10] @ inertial_payload.psi[0]
                expected_symmetry += S[:, 10:] @ inertial_payload.psi[1]
                np.testing.assert_allclose(
                    symmetry_data.r[:10], expected_symmetry[:10], atol=3e-5
                )
                np.testing.assert_allclose(
                    symmetry_data.r[10:], expected_symmetry[10:], atol=3e-5
                )
                for i in range(2):
                    np.testing.assert_allclose(
                        symmetry_data.Rp[:, 3 + 10 * i : 13 + 10 * i],
                        S[:, 10 * i : 10 * (i + 1)] @ inertial_payload.dpsi_dp[i],
                        atol=3e-5,
                    )

                selected_mass = float(psi[0] + psi[10])
                complete_mass = self.unselected_mass + selected_mass
                self.assertGreater(abs(complete_mass - selected_mass), 1e-3)
                total_mass = module.ResidualModelTotalMass(
                    state,
                    complete_mass - 0.25,
                    fixture["actuation"].nu,
                    np_total,
                    "z_inertial",
                )
                total_mass_data = total_mass.createData(manager_data)
                self.assertEqual(total_mass_data.np_offset, 3)
                total_mass.calc(total_mass_data, x, u)
                total_mass.calcDiff(total_mass_data, x, u)
                self.assertAlmostEqual(float(total_mass_data.r[0]), 0.25, places=4)
                for i in range(2):
                    np.testing.assert_allclose(
                        np.asarray(total_mass_data.Rp).reshape(1, -1)[
                            :, 3 + 10 * i : 13 + 10 * i
                        ],
                        np.asarray(inertial_payload.dpsi_dp[i], dtype=dtype)[0:1, :],
                        atol=3e-5,
                    )

                copied_data = copy.copy(inertial_data)
                copied_data.Rp = np.zeros_like(copied_data.Rp)
                self.assertFalse(np.array_equal(copied_data.Rp, inertial_data.Rp))
                copied_model = copy.copy(symmetry)
                np.testing.assert_allclose(copied_model.S, symmetry.S)
                casted = inertial.cast(cast_dtype)
                self.assertEqual(casted.np, inertial.np)

                manager.changeParamStatus("a_actuation", False)
                manager_data.resize(manager)
                with self.assertRaises(crocoddyl.Exception):
                    actuation.calc(actuation_data, x, u)
                manager.changeParamStatus("a_actuation", True)
                manager_data.resize(manager)
                manager.update(manager_data, p)
                actuation.calcDiff(actuation_data, x, u)

                with self.assertRaises(crocoddyl.Exception):
                    module.ResidualModelSymmetryParameters(
                        state,
                        np.zeros((2, 3), dtype=dtype),
                        fixture["actuation"].nu,
                        np_total,
                        "z_inertial",
                    )
                with self.assertRaises(crocoddyl.Exception):
                    module.ResidualModelSymmetryParameters(
                        state,
                        np.eye(np_total - 1, dtype=dtype),
                        fixture["actuation"].nu,
                        np_total,
                    )
                with self.assertRaises(crocoddyl.Exception):
                    module.ResidualModelTotalMass(
                        module.StateVector(4), 0.0, 2, np_total, "z_inertial"
                    )
                with self.assertRaises(crocoddyl.Exception):
                    module.ResidualModelInertialParameters(
                        state,
                        np.zeros(20, dtype=dtype),
                        fixture["actuation"].nu,
                        np_total,
                        "missing",
                    ).createData(manager_data)

    def test_energy_and_power_bindings(self):
        for module, dtype, state, cast_dtype in self.scalar_cases():
            with self.subTest(module=module.__name__):
                fixture = self.make_fixture(
                    module, dtype, state, self.joint_nq, self.body_names
                )
                shared = fixture["shared"]
                x, u = fixture["x"], fixture["u"]
                np_total = fixture["manager"].np
                tolerance = 5e-3 if dtype == np.float32 else 1e-9

                potential = module.ResidualModelPotentialEnergy(
                    state, fixture["actuation"].nu, np_total, 0.3, "z_inertial"
                )
                potential_data = potential.createData(shared)
                potential.calc(potential_data, x, u)
                potential.calcDiff(potential_data, x, u)
                self.assertTrue(np.all(np.isfinite(potential_data.r)))
                self.assertTrue(np.all(np.isfinite(potential_data.Rx)))
                self.assertTrue(np.all(np.isfinite(potential_data.Rp)))
                running = potential_data.r.copy()
                potential.calc(potential_data, x)
                np.testing.assert_allclose(potential_data.r, running, atol=tolerance)

                kinetic = module.ResidualModelKineticEnergy(
                    state, fixture["actuation"].nu, np_total, 0.4, "z_inertial"
                )
                kinetic_data = kinetic.createData(shared)
                kinetic.calc(kinetic_data, x, u)
                kinetic.calcDiff(kinetic_data, x, u)
                self.assertTrue(np.all(np.isfinite(kinetic_data.r)))
                self.assertTrue(np.all(np.isfinite(kinetic_data.Rx)))
                self.assertTrue(np.all(np.isfinite(kinetic_data.Rp)))
                running = kinetic_data.r.copy()
                kinetic.calc(kinetic_data, x)
                np.testing.assert_allclose(kinetic_data.r, running, atol=tolerance)

                power = module.ResidualModelPower(
                    state,
                    fixture["actuation"].nu,
                    np_total,
                    0.025,
                    "z_inertial",
                    "a_actuation",
                )
                power_data = power.createData(shared)
                power.calc(power_data, x)
                power.calcDiff(power_data, x)
                np.testing.assert_array_equal(power_data.r, np.zeros(1, dtype=dtype))
                self.assertTrue(np.all(power_data.Rx == dtype(0)))
                self.assertTrue(np.all(power_data.Rp == dtype(0)))
                power_data.r[:] = dtype(1)
                power_data.Rx[:] = dtype(1)
                power_data.Ru[:] = dtype(1)
                power_data.Rp[:] = dtype(1)
                power.calc(power_data, x, u)
                power.calcDiff(power_data, x, u)
                np.testing.assert_array_equal(power_data.r, np.zeros(1, dtype=dtype))
                self.assertTrue(np.all(power_data.Rx == dtype(0)))
                self.assertTrue(np.all(power_data.Ru == dtype(0)))
                self.assertTrue(np.all(power_data.Rp == dtype(0)))

                potential.reference = 0.7
                kinetic.reference = 0.8
                power.reference = 0.9
                self.assertAlmostEqual(float(potential.reference), 0.7, places=5)
                self.assertAlmostEqual(float(kinetic.reference), 0.8, places=5)
                self.assertAlmostEqual(float(power.reference), 0.9, places=5)
                self.assertEqual(power.inertial_param_name, "z_inertial")
                self.assertEqual(power.actuation_param_name, "a_actuation")
                self.assertEqual(potential.cast(cast_dtype).np, np_total)
                self.assertIsInstance(copy.copy(kinetic_data), type(kinetic_data))

                plain = module.ResidualModelPotentialEnergy(
                    state, fixture["actuation"].nu, 0
                )
                self.assertEqual(plain.param_name, "")
                with self.assertRaises(TypeError):
                    module.ResidualModelPotentialEnergy(
                        state, fixture["actuation"].nu, np_total, 0.0, None
                    )
                with self.assertRaises(crocoddyl.Exception):
                    module.ResidualModelKineticEnergy(module.StateVector(4), 2, 0)
                with self.assertRaises(crocoddyl.Exception):
                    module.ResidualModelPotentialEnergy(
                        state,
                        fixture["actuation"].nu,
                        np_total,
                        0.0,
                        "a_actuation",
                    ).createData(shared)

                fixture["manager"].changeParamStatus("z_inertial", False)
                fixture["manager_data"].resize(fixture["manager"])
                with self.assertRaises(crocoddyl.Exception):
                    potential.calc(potential_data, x, u)
                fixture["manager"].changeParamStatus("z_inertial", True)
                fixture["manager_data"].resize(fixture["manager"])
                fixture["manager"].update(fixture["manager_data"], fixture["p"])
                potential.calc(potential_data, x, u)


if __name__ == "__main__":
    unittest.main()
