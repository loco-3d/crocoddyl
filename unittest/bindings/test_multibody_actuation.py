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


def make_joint_probe(module, dtype, joint_id=3):
    class JointProbe(module.JointDynamicsModelAbstract):
        def __init__(self):
            super().__init__(joint_id, 1, 1, 1)
            self.parameters = np.array([dtype(0.4)], dtype=dtype)
            self.create_calls = 0
            self.update_calls = 0
            self.regressor_calls = 0

        def calc(self, data, q, v, u):
            data.friction = np.array([dtype(0.25)], dtype=dtype)
            data.tau = np.asarray(u, dtype=dtype) - data.friction

        def calcDiff(self, data, q, v, u):
            data.dtau_dq = np.zeros((1, 1), dtype=dtype)
            data.dtau_dv = np.array([[-dtype(0.5)]], dtype=dtype)
            data.dtau_du = np.eye(1, dtype=dtype)
            data.Mtau = np.eye(1, dtype=dtype)

        def commands(self, data, q, v, tau):
            data.u = np.asarray(tau, dtype=dtype) + dtype(0.25)

        def get_np(self):
            return 1

        def set_parameters(self, p):
            self.parameters = np.asarray(p, dtype=dtype).copy()

        def get_parameters(self):
            return self.parameters.copy()

        def get_parametrization(self):
            return self.parameters.copy()

        def updateParametrizationDerivative(self, dgamma_dp):
            self.update_calls += 1
            return np.array([[dtype(2.0)]], dtype=dtype)

        def computeJointTorqueRegressor(self, joint_dtau_dp, q, v, u):
            self.regressor_calls += 1
            return np.array([[np.asarray(u, dtype=dtype)[0]]], dtype=dtype)

        def createData(self):
            self.create_calls += 1
            return module.JointDynamicsDataAbstract(self)

    return JointProbe()


class MultibodyActuationBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state64 = crocoddyl.StateMultibody(
            pinocchio.buildSampleModelHumanoidRandom()
        )
        cls.state32 = (
            cls.state64.cast(crocoddyl.DType.Float32)
            if PINOCCHIO_FLOAT32_AVAILABLE
            else None
        )
        one_dof_model = pinocchio.Model()
        cls.one_dof_joint_id = one_dof_model.addJoint(
            0, pinocchio.JointModelRY(), pinocchio.SE3.Identity(), "joint"
        )
        one_dof_model.appendBodyToJoint(
            cls.one_dof_joint_id,
            pinocchio.Inertia.Random(),
            pinocchio.SE3.Identity(),
        )
        cls.one_dof_state64 = crocoddyl.StateMultibody(one_dof_model)
        cls.one_dof_state32 = (
            cls.one_dof_state64.cast(crocoddyl.DType.Float32)
            if PINOCCHIO_FLOAT32_AVAILABLE
            else None
        )
        pin_model = cls.state64.pinocchio
        cls.root_id = pin_model.getJointId("root_joint")
        cls.first_id = cls.root_id + 1
        first_joint = pin_model.joints[cls.first_id]
        cls.first_nq = first_joint.nq
        cls.first_nv = first_joint.nv
        cls.first_idx_v = first_joint.idx_v
        poses = (
            pinocchio.SE3(np.eye(3), np.array([0.15, 0.0, 0.0])),
            pinocchio.SE3(np.eye(3), np.array([0.0, 0.15, 0.0])),
            pinocchio.SE3(np.eye(3), np.array([-0.15, 0.0, 0.0])),
            pinocchio.SE3(np.eye(3), np.array([0.0, -0.15, 0.0])),
        )
        signs = (
            crocoddyl.ThrusterType.CCW,
            crocoddyl.ThrusterType.CW,
            crocoddyl.ThrusterType.CCW,
            crocoddyl.ThrusterType.CW,
        )
        cls.thrusters64 = [
            crocoddyl.Thruster(pose, 0.1 + 0.01 * i, signs[i], 0.1, 5.0)
            for i, pose in enumerate(poses)
        ]
        cls.thrusters32 = (
            [thruster.cast(crocoddyl.DType.Float32) for thruster in cls.thrusters64]
            if PINOCCHIO_FLOAT32_AVAILABLE
            else None
        )

    def scalar_cases(self):
        cases = [(crocoddyl, np.float64, self.state64, self.thrusters64)]
        if PINOCCHIO_FLOAT32_AVAILABLE:
            cases.append(
                (crocoddyl_float32, np.float32, self.state32, self.thrusters32)
            )
        return cases

    def test_abstract_dispatch_fallback_and_errors(self):
        for module, dtype, state, _ in self.scalar_cases():
            with self.subTest(module=module.__name__):
                model = make_joint_probe(module, dtype, self.first_id)
                data = model.createData()
                self.assertEqual(model.create_calls, 1)
                self.assertIsInstance(data, module.JointDynamicsDataAbstract)
                q = np.zeros(1, dtype=dtype)
                v = np.array([dtype(0.2)], dtype=dtype)
                u = np.array([dtype(0.7)], dtype=dtype)
                model.calc(data, q, v, u)
                model.calcDiff(data, q, v, u)
                self.assertTrue(np.allclose(data.tau, [0.45]))
                self.assertTrue(np.allclose(data.dtau_dv, [[-0.5]]))
                model.commands(data, q, v, np.array([dtype(0.3)], dtype=dtype))
                self.assertTrue(np.allclose(data.u, [0.55]))

                dgamma = np.zeros((1, 1), dtype=dtype)
                self.assertTrue(
                    np.allclose(model.updateParametrizationDerivative(dgamma), [[2.0]])
                )
                self.assertEqual(model.update_calls, 1)
                regressor = np.zeros((1, 1), dtype=dtype)
                self.assertTrue(
                    np.allclose(
                        model.computeJointTorqueRegressor(regressor, q, v, u),
                        [[0.7]],
                    )
                )
                self.assertEqual(model.regressor_calls, 1)
                self.assertFalse(hasattr(model, "_updateParametrizationDerivative"))

                actuation = module.ActuationModelMultibody(state, [model])
                actuation_data = actuation.createData()
                x = state.rand()
                actuation.calc(actuation_data, x, u)
                actuation.calcDiff(actuation_data, x, u)
                self.assertTrue(
                    np.allclose(actuation_data.tau[self.first_idx_v], dtype(0.45))
                )
                params = module.ActuationMultibodyParams(actuation)
                params_data = params.createData()
                params.update(params_data, np.array([dtype(0.6)], dtype=dtype))
                self.assertEqual(model.update_calls, 2)
                joint_regressor = np.zeros((state.nv, 1), dtype=dtype)
                actuation.computeJointTorqueRegressor(
                    joint_regressor, x, np.array([dtype(0.7)], dtype=dtype)
                )
                self.assertEqual(model.regressor_calls, 2)
                self.assertTrue(
                    np.allclose(joint_regressor[self.first_idx_v], dtype(0.7))
                )

                base_data = module.JointDynamicsModelAbstract.createData(model)
                self.assertIsInstance(base_data, module.JointDynamicsDataAbstract)
                self.assertEqual(model.create_calls, 2)
                module.JointDynamicsModelAbstract.updateParametrizationDerivative(
                    model, np.zeros((0, 0), dtype=dtype)
                )
                self.assertEqual(model.update_calls, 2)

                fallback = module.JointDynamicsModelAbstract(4, 1, 1, 1)
                fallback_data = fallback.createData()
                self.assertIsInstance(fallback_data, module.JointDynamicsDataAbstract)
                fallback_matrix = np.ones((0, 0), dtype=dtype)
                fallback.updateParametrizationDerivative(fallback_matrix)
                with self.assertRaises(crocoddyl.Exception):
                    fallback.calc(
                        fallback_data,
                        np.zeros(1, dtype=dtype),
                        np.zeros(1, dtype=dtype),
                        np.zeros(1, dtype=dtype),
                    )
                with self.assertRaises(crocoddyl.Exception):
                    module.JointDynamicsDataAbstract(None)

    def test_identity_and_all_friction_modes(self):
        friction_modes = (
            ("COULOMB", 2, 0),
            ("VISCOUS", 1, 0),
            ("STRIBECK", 2, 0),
            ("COULOMB_VISCOUS", 3, 0),
            ("COULOMB_STRIBECK", 4, 0),
            ("VISCOUS_STRIBECK", 3, 0),
            ("FULL", 6, 0),
            ("COULOMB_FIXED_SMOOTHING", 1, 1),
            ("STRIBECK_FIXED_SMOOTHING", 0, 2),
            ("COULOMB_VISCOUS_FIXED_SMOOTHING", 2, 1),
            ("COULOMB_STRIBECK_FIXED_SMOOTHING", 1, 3),
            ("VISCOUS_STRIBECK_FIXED_SMOOTHING", 1, 2),
            ("FULL_FIXED_SMOOTHING", 3, 3),
        )
        for module, dtype, _, _ in self.scalar_cases():
            with self.subTest(module=module.__name__):
                identity = module.JointDynamicsModelIdentity(2, 1, 1)
                data = identity.createData()
                q = np.zeros(1, dtype=dtype)
                v = np.array([dtype(0.3)], dtype=dtype)
                u = np.array([dtype(0.8)], dtype=dtype)
                identity.calc(data, q, v, u)
                identity.calcDiff(data, q, v, u)
                self.assertTrue(np.allclose(data.tau, u))
                self.assertTrue(np.allclose(data.dtau_du, np.eye(1)))
                self.assertTrue(np.allclose(data.Mtau, np.eye(1)))

                for name, np_, ns in friction_modes:
                    mode = getattr(crocoddyl.JointFrictionType, name)
                    mu = np.linspace(0.1, 0.1 * max(1, np_), np_, dtype=dtype)
                    fixed = np.linspace(0.2, 0.2 * max(1, ns), ns, dtype=dtype)
                    if ns:
                        model = module.JointDynamicsModelFriction(
                            3, 1, mu, mode, False, fixed
                        )
                    else:
                        model = module.JointDynamicsModelFriction(3, 1, mu, mode, False)
                    self.assertEqual(model.np, np_)
                    model_data = model.createData()
                    model.calc(model_data, q, v, u)
                    model.calcDiff(model_data, q, v, u)
                    passive = (
                        module.JointDynamicsModelFriction(3, 1, mu, mode, True, fixed)
                        if ns
                        else module.JointDynamicsModelFriction(3, 1, mu, mode, True)
                    )
                    self.assertTrue(passive.passive)
                    self.assertEqual(passive.nu, 0)

                with self.assertRaises(crocoddyl.Exception):
                    module.JointDynamicsModelFriction(
                        3,
                        1,
                        np.zeros(1, dtype=dtype),
                        crocoddyl.JointFrictionType.FULL,
                    )

    def test_c_order_mutable_regressors(self):
        cases = [(crocoddyl, np.float64, self.one_dof_state64)]
        if PINOCCHIO_FLOAT32_AVAILABLE:
            cases.append((crocoddyl_float32, np.float32, self.one_dof_state32))
        for module, dtype, state in cases:
            with self.subTest(module=module.__name__):
                p = np.array([1.3, 2.0, 5.0, 0.2, 4.0, np.log(0.5)], dtype=dtype)
                model = module.JointDynamicsModelFriction(
                    self.one_dof_joint_id,
                    1,
                    p,
                    crocoddyl.JointFrictionType.FULL,
                )
                q = np.array([0.1], dtype=dtype)
                v = np.array([0.37], dtype=dtype)
                u = np.array([0.61], dtype=dtype)
                tanh1v = np.tanh(p[1] * v[0])
                tanh2v = np.tanh(p[2] * v[0])
                tanh4v = np.tanh(p[4] * v[0])
                expected = np.array(
                    [
                        tanh1v - tanh2v,
                        p[0] * (1.0 - tanh1v**2) * v[0],
                        -p[0] * (1.0 - tanh2v**2) * v[0],
                        tanh4v,
                        p[3] * (1.0 - tanh4v**2) * v[0],
                        np.exp(p[5]) * v[0],
                    ],
                    dtype=dtype,
                ).reshape(1, -1)

                direct = np.full((1, 6), np.nan, dtype=dtype, order="C")
                model.computeJointTorqueRegressor(direct, q, v, u)
                np.testing.assert_allclose(direct, expected, rtol=2e-5, atol=2e-6)

                dgamma = np.full((6, 6), np.nan, dtype=dtype, order="C")
                model.updateParametrizationDerivative(dgamma)
                expected_dgamma = np.eye(6, dtype=dtype)
                expected_dgamma[-1, -1] = np.exp(p[-1])
                np.testing.assert_allclose(
                    dgamma, expected_dgamma, rtol=2e-5, atol=2e-6
                )

                actuation = module.ActuationModelMultibody(state, [model])
                x = np.array([q[0], v[0]], dtype=dtype)
                assembled = np.full((1, 6), np.nan, dtype=dtype, order="C")
                actuation.computeJointTorqueRegressor(assembled, x, u)
                np.testing.assert_allclose(assembled, expected, rtol=2e-5, atol=2e-6)

                assembled_dgamma = np.full((6, 6), np.nan, dtype=dtype, order="C")
                actuation.updateParametrizationDerivative(assembled_dgamma)
                np.testing.assert_allclose(
                    assembled_dgamma, expected_dgamma, rtol=2e-5, atol=2e-6
                )
                gamma = np.full(6, np.nan, dtype=dtype)
                actuation.update_p(p, gamma)
                expected_gamma = p.copy()
                expected_gamma[-1] = np.exp(p[-1])
                np.testing.assert_allclose(gamma, expected_gamma, rtol=2e-5, atol=2e-6)

    def test_thruster_mapping_bounds_cast_copy_and_two_data_refresh(self):
        for module, dtype, _, thrusters in self.scalar_cases():
            with self.subTest(module=module.__name__):
                model = module.JointDynamicsModelThruster(thrusters)
                data_a = model.createData()
                data_b = model.createData()
                q = np.zeros(7, dtype=dtype)
                v = np.zeros(6, dtype=dtype)
                u = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
                model.calc(data_a, q, v, u)
                model.calc(data_b, q, v, u)
                old_mapping = data_a.dtau_du.copy()

                parameters = np.array([0.2, 0.3, 0.4, 0.5], dtype=dtype)
                model.set_parameters(parameters)
                model.calc(data_a, q, v, u)
                model.calc(data_b, q, v, u)
                model.calcDiff(data_a, q, v, u)
                model.calcDiff(data_b, q, v, u)
                self.assertFalse(np.allclose(old_mapping, data_a.dtau_du))
                self.assertTrue(np.allclose(data_a.dtau_du, model.Wthrust))
                self.assertTrue(np.allclose(data_b.dtau_du, model.Wthrust))
                self.assertTrue(np.allclose(data_a.Mtau, data_b.Mtau))
                regressor = np.zeros((6, 4), dtype=dtype)
                model.computeJointTorqueRegressor(regressor, q, v, u)
                expected_regressor = np.zeros((6, 4), dtype=dtype)
                expected_regressor[5] = [-1.0, 2.0, -3.0, 4.0]
                np.testing.assert_allclose(regressor, expected_regressor)
                dgamma = np.full((4, 4), np.nan, dtype=dtype, order="C")
                model.updateParametrizationDerivative(dgamma)
                np.testing.assert_allclose(dgamma, np.eye(4, dtype=dtype))
                copied_model = copy.copy(model)
                copied_data = copy.copy(data_a)
                copied_data.tau = np.ones(6, dtype=dtype)
                self.assertFalse(np.allclose(copied_data.tau, data_a.tau))
                self.assertTrue(np.allclose(copied_model.get_parameters(), parameters))
                self.assertEqual(model.thrusters[0].min_thrust, dtype(0.1))
                self.assertEqual(model.thrusters[0].max_thrust, dtype(5.0))

        casted = crocoddyl.JointDynamicsModelThruster(self.thrusters64).cast(
            crocoddyl.DType.Float32
        )
        self.assertIsInstance(casted, crocoddyl_float32.JointDynamicsModelThruster)
        self.assertTrue(np.allclose(casted.get_parameters(), [0.1, 0.11, 0.12, 0.13]))

    def test_multibody_offsets_parameters_manager_and_failures(self):
        for module, dtype, state, thrusters in self.scalar_cases():
            with self.subTest(module=module.__name__):
                identity = module.JointDynamicsModelIdentity(
                    self.first_id, self.first_nq, self.first_nv
                )
                thruster = module.JointDynamicsModelThruster(thrusters)
                actuation = module.ActuationModelMultibody(state, [thruster, identity])
                self.assertEqual(actuation.nu, 4 + self.first_nv)
                self.assertEqual(actuation.np, 4)
                data = actuation.createData()
                self.assertIsInstance(data, module.ActuationDataMultibody)
                self.assertEqual(len(data.joint[self.root_id]), 1)
                self.assertEqual(len(data.joint[self.first_id]), 1)
                x = state.rand()
                u = np.linspace(
                    dtype(0.1), dtype(0.1 * actuation.nu), actuation.nu
                ).astype(dtype)
                actuation.calc(data, x, u)
                actuation.calcDiff(data, x, u)
                self.assertTrue(
                    np.allclose(
                        data.tau[self.first_idx_v : self.first_idx_v + self.first_nv],
                        u[4:],
                    )
                )

                params = module.ActuationMultibodyParams(actuation)
                params_data = params.createData()
                self.assertIsInstance(params_data, module.ActuationMultibodyParamsData)
                self.assertIsInstance(params_data, module.DynamicsParamsDataAbstract)
                self.assertIsInstance(params_data, module.ParamsDataAbstract)
                self.assertEqual(
                    (params_data.np_action, params_data.np_dynamics),
                    (0, 4),
                )
                p = np.array([0.21, 0.31, 0.41, 0.51], dtype=dtype)
                params.update(params_data, p)
                self.assertTrue(np.allclose(params_data.p, p))
                self.assertTrue(np.allclose(params_data.gamma, p))

                manager = module.ParameterManager(state)
                manager.addParam("actuation", params)
                manager_data = manager.createData()
                manager.update(manager_data, p)
                self.assertEqual(manager.np, 4)
                manager.changeParamStatus("actuation", False)
                self.assertEqual(manager.np, 0)
                manager.changeParamStatus("actuation", True)
                self.assertEqual(manager.np, 4)

                params_copy = copy.copy(params)
                data_copy = copy.copy(params_data)
                data_copy.gamma = np.zeros(4, dtype=dtype)
                self.assertFalse(np.allclose(data_copy.gamma, params_data.gamma))
                self.assertIs(params_copy.actuation, actuation)

                with self.assertRaises(crocoddyl.Exception):
                    module.ActuationMultibodyParams(None)
                with self.assertRaises(crocoddyl.Exception):
                    module.ActuationMultibodyParamsData(None)
                with self.assertRaises(crocoddyl.Exception):
                    module.ActuationModelMultibody(None)
                with self.assertRaises(crocoddyl.Exception):
                    module.ActuationModelMultibody(state, [None])
                with self.assertRaises(crocoddyl.Exception):
                    actuation.calc(data, x, np.zeros(actuation.nu + 1, dtype=dtype))
                with self.assertRaises(crocoddyl.Exception):
                    params.update(params_data, np.zeros(3, dtype=dtype))


if __name__ == "__main__":
    unittest.main()
