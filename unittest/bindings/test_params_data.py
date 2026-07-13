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


class ParamsDataTest(unittest.TestCase):
    def setUp(self):
        self.state = crocoddyl.StateMultibody(
            pinocchio.buildSampleModelHumanoidRandom()
        )

    def test_layout_assignment_resize_active_and_copy(self):
        params = crocoddyl.ParamsDataAbstract(self.state, 2, 3)
        self.assertEqual(params.np, 5)
        self.assertEqual(params.np_action, 2)
        self.assertEqual(params.np_dynamics, 3)
        self.assertEqual(params.p.shape, (5,))
        self.assertEqual(params.dx_dp.shape, (self.state.ndx, 2))
        self.assertEqual(params.dtau_dp.shape, (self.state.nv, 3))
        self.assertTrue(params.active)

        p = np.arange(5.0)
        dx_dp = np.arange(self.state.ndx * 2.0).reshape(self.state.ndx, 2)
        dtau_dp = np.arange(self.state.nv * 3.0).reshape(self.state.nv, 3)
        params.p = p
        params.dx_dp = dx_dp
        params.dtau_dp = dtau_dp
        params.active = False
        self.assertTrue(np.array_equal(params.p, p))
        self.assertTrue(np.array_equal(params.p[: params.np_action], p[:2]))
        self.assertTrue(np.array_equal(params.p[params.np_action :], p[2:]))
        self.assertTrue(np.array_equal(params.dx_dp, dx_dp))
        self.assertTrue(np.array_equal(params.dtau_dp, dtau_dp))

        shallow = copy.copy(params)
        deep = copy.deepcopy(params)
        for copied in (shallow, deep):
            self.assertTrue(np.array_equal(copied.p, params.p))
            self.assertTrue(np.array_equal(copied.dx_dp, params.dx_dp))
            self.assertTrue(np.array_equal(copied.dtau_dp, params.dtau_dp))
            self.assertEqual(copied.np_action, params.np_action)
            self.assertEqual(copied.np_dynamics, params.np_dynamics)
            self.assertFalse(copied.active)

        params.setZero()
        self.assertTrue(np.allclose(params.p, 0.0))
        self.assertTrue(np.allclose(params.dx_dp, 0.0))
        self.assertTrue(np.allclose(params.dtau_dp, 0.0))
        self.assertFalse(params.active)

        params.resize(4, 2)
        self.assertEqual(params.np, 6)
        self.assertEqual(params.np_action, 4)
        self.assertEqual(params.np_dynamics, 2)
        self.assertEqual(params.p.shape, (6,))
        self.assertEqual(params.dx_dp.shape, (self.state.ndx, 4))
        self.assertEqual(params.dtau_dp.shape, (self.state.nv, 2))
        self.assertFalse(params.active)

    def test_model_construction_and_action_inheritance(self):
        model = crocoddyl.ParamsAbstract(self.state, 2)
        action_model = crocoddyl.ActionModelParamsAbstract(self.state, 3)
        model_data = crocoddyl.ParamsDataAbstract(model)
        converted_data = crocoddyl.ParamsDataAbstract(action_model)
        action = crocoddyl.ActionModelParamsDataAbstract(action_model)
        self.assertEqual(
            (model_data.np, model_data.np_action, model_data.np_dynamics),
            (2, 2, 0),
        )
        self.assertEqual(
            (converted_data.np, converted_data.np_action, converted_data.np_dynamics),
            (3, 3, 0),
        )
        self.assertEqual((action.np, action.np_action, action.np_dynamics), (3, 3, 0))
        self.assertEqual(action.dx_dp.shape, (self.state.ndx, 3))
        self.assertEqual(action.dtau_dp.shape, (self.state.nv, 0))
        action.active = False
        action.resize(4, 1)
        self.assertEqual((action.np, action.np_action, action.np_dynamics), (5, 4, 1))
        self.assertFalse(action.active)

        action_copy = copy.copy(action)
        self.assertEqual(action_copy.np_action, 4)
        self.assertEqual(action_copy.np_dynamics, 1)
        with self.assertRaises(TypeError):
            crocoddyl.ActionModelParamsDataAbstract(self.state, 2)

    def test_dynamics_model_construction_inheritance_resize_and_copy(self):
        model = crocoddyl.DynamicsParamsAbstract(self.state, 3)
        dynamics = crocoddyl.DynamicsParamsDataAbstract(model)
        self.assertEqual(
            (dynamics.np, dynamics.np_action, dynamics.np_dynamics), (3, 0, 3)
        )
        self.assertEqual(dynamics.p.shape, (3,))
        self.assertEqual(dynamics.dx_dp.shape, (self.state.ndx, 0))
        self.assertEqual(dynamics.dtau_dp.shape, (self.state.nv, 3))
        self.assertTrue(dynamics.active)

        p = np.array([0.2, 0.4, 0.6])
        dtau_dp = np.arange(self.state.nv * 3.0).reshape(self.state.nv, 3)
        dynamics.p = p
        dynamics.dx_dp = np.empty((self.state.ndx, 0))
        dynamics.dtau_dp = dtau_dp
        dynamics.active = False
        model.update(dynamics, np.zeros(3))
        self.assertTrue(model.checkData(dynamics))
        self.assertTrue(np.array_equal(dynamics.p, p))
        self.assertIs(crocoddyl.DataCollectorParams(dynamics).params, dynamics)

        shallow = copy.copy(dynamics)
        deep = copy.deepcopy(dynamics)
        dynamics.p = np.zeros(3)
        dynamics.dtau_dp = np.zeros((self.state.nv, 3))
        for copied in (shallow, deep):
            self.assertTrue(np.array_equal(copied.p, p))
            self.assertTrue(np.array_equal(copied.dtau_dp, dtau_dp))
            self.assertFalse(copied.active)

        dynamics.resize(0, 4)
        self.assertEqual(
            (dynamics.np, dynamics.np_action, dynamics.np_dynamics), (4, 0, 4)
        )
        self.assertEqual(dynamics.dtau_dp.shape, (self.state.nv, 4))
        self.assertFalse(dynamics.active)
        dynamics.setZero()
        self.assertTrue(np.allclose(dynamics.p, 0.0))
        self.assertTrue(np.allclose(dynamics.dtau_dp, 0.0))
        with self.assertRaises(TypeError):
            crocoddyl.DynamicsParamsDataAbstract(self.state, 3)
        with self.assertRaises(Exception):
            crocoddyl.DynamicsParamsDataAbstract(None)

    def test_core_collector_combinations_and_sharing(self):
        actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        actuation_data = actuation.createData()
        joint_data = crocoddyl.JointDataAbstract(self.state, actuation, actuation.nu)
        params = crocoddyl.ParamsDataAbstract(self.state, 2, 3)
        collectors = [
            (crocoddyl.DataCollectorParams(params), ("params",)),
            (
                crocoddyl.DataCollectorActuationParams(actuation_data, params),
                ("actuation", "params"),
            ),
            (
                crocoddyl.DataCollectorJointParams(joint_data, params),
                ("joint", "params"),
            ),
            (
                crocoddyl.DataCollectorJointActuationParams(
                    actuation_data, joint_data, params
                ),
                ("actuation", "joint", "params"),
            ),
        ]

        self.assertIs(collectors[-1][0].joint, joint_data)

        params.active = False
        for collector, inherited in collectors:
            for field in inherited:
                self.assertTrue(hasattr(collector, field))
            self.assertEqual(collector.params.np, 5)
            self.assertFalse(collector.params.active)
            collector.params.p = np.arange(5.0)
            self.assertTrue(np.array_equal(params.p, np.arange(5.0)))
            copied = copy.copy(collector)
            for field in inherited:
                if field == "pinocchio":
                    collector.pinocchio.M[0, 0] = 42.0
                    self.assertEqual(copied.pinocchio.M[0, 0], 42.0)
                else:
                    self.assertIs(getattr(copied, field), getattr(collector, field))
            self.assertEqual(copied.params.np_action, 2)
            self.assertEqual(copied.params.np_dynamics, 3)

    def test_multibody_contact_and_impulse_collector_combinations(self):
        actuation = crocoddyl.ActuationModelFloatingBase(self.state)
        actuation_data = actuation.createData()
        joint_data = crocoddyl.JointDataAbstract(self.state, actuation, actuation.nu)
        params = crocoddyl.ParamsDataAbstract(self.state, 2, 3)
        pinocchio_data = self.state.pinocchio.createData()
        contacts = crocoddyl.ContactModelMultiple(self.state, actuation.nu)
        impulses = crocoddyl.ImpulseModelMultiple(self.state)
        contact_data = contacts.createData(pinocchio_data)
        impulse_data = impulses.createData(pinocchio_data)
        collectors = [
            (
                crocoddyl.DataCollectorMultibodyParams(pinocchio_data, params),
                ("pinocchio", "params"),
            ),
            (
                crocoddyl.DataCollectorActMultibodyParams(
                    pinocchio_data, actuation_data, params
                ),
                ("pinocchio", "actuation", "params"),
            ),
            (
                crocoddyl.DataCollectorJointActMultibodyParams(
                    pinocchio_data, actuation_data, joint_data, params
                ),
                ("pinocchio", "actuation", "joint", "params"),
            ),
            (
                crocoddyl.DataCollectorMultibodyInContactParams(
                    pinocchio_data, contact_data, params
                ),
                ("pinocchio", "contacts", "params"),
            ),
            (
                crocoddyl.DataCollectorActMultibodyInContactParams(
                    pinocchio_data, actuation_data, contact_data, params
                ),
                ("pinocchio", "actuation", "contacts", "params"),
            ),
            (
                crocoddyl.DataCollectorJointActMultibodyInContactParams(
                    pinocchio_data,
                    actuation_data,
                    joint_data,
                    contact_data,
                    params,
                ),
                ("pinocchio", "actuation", "joint", "contacts", "params"),
            ),
            (
                crocoddyl.DataCollectorMultibodyInImpulseParams(
                    pinocchio_data, impulse_data, params
                ),
                ("pinocchio", "impulses", "params"),
            ),
        ]

        for collector, inherited in collectors:
            for field in inherited:
                self.assertTrue(hasattr(collector, field))
            self.assertEqual(collector.params.np, 5)
            collector.params.active = False
            self.assertFalse(params.active)
            copied = copy.copy(collector)
            for field in inherited:
                if field == "pinocchio":
                    collector.pinocchio.M[0, 0] = 42.0
                    self.assertEqual(copied.pinocchio.M[0, 0], 42.0)
                else:
                    self.assertIs(getattr(copied, field), getattr(collector, field))
            self.assertEqual(copied.params.np, params.np)

    def test_float32_layout_assignment_and_copy(self):
        state = crocoddyl_float32.StateVector(4)
        params = crocoddyl_float32.ParamsDataAbstract(state, 2, 3)
        params.p = np.arange(5, dtype=np.float32)
        params.dx_dp = np.ones((state.ndx, 2), dtype=np.float32)
        params.dtau_dp = np.full((state.nv, 3), 2.0, dtype=np.float32)
        params.active = False
        copied = copy.copy(params)
        self.assertEqual(copied.p.dtype, np.float32)
        self.assertTrue(np.array_equal(copied.p, params.p))
        self.assertTrue(np.array_equal(copied.dx_dp, params.dx_dp))
        self.assertTrue(np.array_equal(copied.dtau_dp, params.dtau_dp))
        self.assertFalse(copied.active)

        model = crocoddyl_float32.ActionModelParamsAbstract(state, 2)
        action = crocoddyl_float32.ActionModelParamsDataAbstract(model)
        self.assertEqual(action.dx_dp.shape, (state.ndx, 2))
        self.assertEqual(action.dtau_dp.shape, (state.nv, 0))
        collector = crocoddyl_float32.DataCollectorParams(params)
        self.assertEqual(collector.params.np, 5)

        dynamics_model = crocoddyl_float32.DynamicsParamsAbstract(state, 3)
        dynamics = crocoddyl_float32.DynamicsParamsDataAbstract(dynamics_model)
        self.assertEqual(
            (dynamics.np, dynamics.np_action, dynamics.np_dynamics), (3, 0, 3)
        )
        dynamics.p = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        dynamics.dtau_dp = np.arange(state.nv * 3, dtype=np.float32).reshape(
            state.nv, 3
        )
        copied_dynamics = copy.deepcopy(dynamics)
        dynamics.p = np.zeros(3, dtype=np.float32)
        self.assertTrue(
            np.array_equal(
                copied_dynamics.p, np.array([0.2, 0.4, 0.6], dtype=np.float32)
            )
        )
        dynamics_model.update(copied_dynamics, np.zeros(3, dtype=np.float32))
        self.assertTrue(dynamics_model.checkData(copied_dynamics))
        self.assertIs(
            crocoddyl_float32.DataCollectorParams(copied_dynamics).params,
            copied_dynamics,
        )


if __name__ == "__main__":
    unittest.main()
