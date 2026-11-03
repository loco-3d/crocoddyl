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
        params = crocoddyl.ParamsDataAbstract(2, 3)
        self.assertEqual(params.np, 5)
        self.assertEqual(params.np_action, 2)
        self.assertEqual(params.np_dynamics, 3)
        self.assertEqual(params.p.shape, (5,))
        self.assertTrue(params.active)

        p = np.arange(5.0)
        params.p = p
        params.active = False
        self.assertTrue(np.array_equal(params.p, p))
        self.assertTrue(np.array_equal(params.p[: params.np_action], p[:2]))
        self.assertTrue(np.array_equal(params.p[params.np_action :], p[2:]))

        shallow = copy.copy(params)
        deep = copy.deepcopy(params)
        for copied in (shallow, deep):
            self.assertTrue(np.array_equal(copied.p, params.p))
            self.assertEqual(copied.np_action, params.np_action)
            self.assertEqual(copied.np_dynamics, params.np_dynamics)
            self.assertFalse(copied.active)

        params.setZero()
        self.assertTrue(np.allclose(params.p, 0.0))
        self.assertFalse(params.active)

        params.resize(4, 2)
        self.assertEqual(params.np, 6)
        self.assertEqual(params.np_action, 4)
        self.assertEqual(params.np_dynamics, 2)
        self.assertEqual(params.p.shape, (6,))
        self.assertFalse(params.active)

    def test_model_construction_and_action_inheritance(self):
        model = crocoddyl.ParamsAbstract(self.state, 2)
        action_model = crocoddyl.ActionModelParamsAbstract(self.state, 3)
        model_data = crocoddyl.ParamsDataAbstract(model.np)
        converted_data = crocoddyl.ParamsDataAbstract(action_model.np)
        empty_action = crocoddyl.ActionModelParamsDataAbstract()
        action = crocoddyl.ActionModelParamsDataAbstract(action_model.np)
        self.assertEqual((empty_action.np, empty_action.np_action), (0, 0))
        self.assertEqual(
            (model_data.np, model_data.np_action, model_data.np_dynamics),
            (2, 2, 0),
        )
        self.assertEqual(
            (converted_data.np, converted_data.np_action, converted_data.np_dynamics),
            (3, 3, 0),
        )
        self.assertEqual((action.np, action.np_action, action.np_dynamics), (3, 3, 0))
        action.active = False
        action.resize(4, 1)
        self.assertEqual((action.np, action.np_action, action.np_dynamics), (5, 4, 1))
        self.assertFalse(action.active)

        action_copy = copy.copy(action)
        self.assertEqual(action_copy.np_action, 4)
        self.assertEqual(action_copy.np_dynamics, 1)

    def test_dynamics_model_construction_inheritance_resize_and_copy(self):
        model = crocoddyl.DynamicsParamsAbstract(self.state, 3)
        empty_dynamics = crocoddyl.DynamicsParamsDataAbstract()
        dynamics = crocoddyl.DynamicsParamsDataAbstract(model.np)
        self.assertEqual((empty_dynamics.np, empty_dynamics.np_dynamics), (0, 0))
        self.assertEqual(
            (dynamics.np, dynamics.np_action, dynamics.np_dynamics), (3, 0, 3)
        )
        self.assertEqual(dynamics.p.shape, (3,))
        self.assertTrue(dynamics.active)

        p = np.array([0.2, 0.4, 0.6])
        dynamics.p = p
        dynamics.active = False
        model.update(dynamics, np.zeros(3))
        self.assertTrue(model.checkData(dynamics))
        self.assertTrue(np.array_equal(dynamics.p, p))
        self.assertIs(crocoddyl.DataCollectorParams(dynamics).params, dynamics)

        shallow = copy.copy(dynamics)
        deep = copy.deepcopy(dynamics)
        dynamics.p = np.zeros(3)
        for copied in (shallow, deep):
            self.assertTrue(np.array_equal(copied.p, p))
            self.assertFalse(copied.active)

        dynamics.resize(0, 4)
        self.assertEqual(
            (dynamics.np, dynamics.np_action, dynamics.np_dynamics), (4, 0, 4)
        )
        self.assertFalse(dynamics.active)
        dynamics.setZero()
        self.assertTrue(np.allclose(dynamics.p, 0.0))

    def test_core_collector_combinations_and_sharing(self):
        actuation = crocoddyl.ActuationModelMultibody(self.state)
        actuation_data = actuation.createData()
        joint_data = crocoddyl.JointDataAbstract(self.state, actuation, actuation.nu)
        params = crocoddyl.ParamsDataAbstract(2, 3)
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

    def test_multibody_collector_combinations(self):
        actuation = crocoddyl.ActuationModelMultibody(self.state)
        actuation_data = actuation.createData()
        joint_data = crocoddyl.JointDataAbstract(self.state, actuation, actuation.nu)
        params = crocoddyl.ParamsDataAbstract(2, 3)
        pinocchio_data = self.state.pinocchio.createData()
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
            self.assertEqual(copied.params.np_action, 2)
            self.assertEqual(copied.params.np_dynamics, 3)

    def test_float32_layout_assignment_and_copy(self):
        state = crocoddyl_float32.StateVector(4)
        params = crocoddyl_float32.ParamsDataAbstract(2, 3)
        params.p = np.arange(5, dtype=np.float32)
        params.active = False
        copied = copy.copy(params)
        self.assertEqual(copied.p.dtype, np.float32)
        self.assertTrue(np.array_equal(copied.p, params.p))
        self.assertFalse(copied.active)

        model = crocoddyl_float32.ActionModelParamsAbstract(state, 2)
        self.assertIsInstance(
            crocoddyl_float32.ActionModelParamsDataAbstract(model.np),
            crocoddyl_float32.ActionModelParamsDataAbstract,
        )
        collector = crocoddyl_float32.DataCollectorParams(params)
        self.assertEqual(collector.params.np, 5)

        dynamics_model = crocoddyl_float32.DynamicsParamsAbstract(state, 3)
        dynamics = crocoddyl_float32.DynamicsParamsDataAbstract(dynamics_model.np)
        self.assertEqual(
            (dynamics.np, dynamics.np_action, dynamics.np_dynamics), (3, 0, 3)
        )
        dynamics.p = np.array([0.2, 0.4, 0.6], dtype=np.float32)
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
