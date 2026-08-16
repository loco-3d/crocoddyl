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
import pinocchio.float32

import crocoddyl
import crocoddyl.float32 as crocoddyl_float32


def make_probe(module, state, dtype, nc, nu):
    class Probe(module.ImplicitConstraintModelAbstract):
        def __init__(self):
            super().__init__(state, pinocchio.LOCAL, nc, nu)
            self.calc_calls = 0
            self.diff_calls = 0
            self.force_calls = 0

        def calc(self, data, x):
            self.calc_calls += 1
            data.Jc = np.full((nc, state.nv), dtype(2), dtype=dtype)
            data.a0 = np.arange(nc, dtype=dtype) + dtype(1)

        def calcDiff(self, data, x):
            self.diff_calls += 1
            data.da0_dx = np.full((nc, state.ndx), dtype(3), dtype=dtype)
            data.dv0_dq = np.full((nc, state.nv), dtype(4), dtype=dtype)

        def updateForce(self, data, force):
            self.force_calls += 1
            vector = np.zeros(6, dtype=dtype)
            vector[: min(3, nc)] = force[: min(3, nc)]
            data.f = type(data.f)(vector)
            data.fext = type(data.fext)(vector)

    return Probe()


class ImplicitConstraintsBindingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state64 = crocoddyl.StateMultibody(
            pinocchio.buildSampleModelHumanoidRandom()
        )
        cls.state32 = cls.state64.cast(crocoddyl.DType.Float32)

    def check_scalar(self, module, dtype, state):
        tolerance = 1e-5 if dtype == np.float32 else 1e-12
        pin_data = state.pinocchio.createData()
        probe = make_probe(module, state, dtype, 3, state.nv)
        data = probe.createData(pin_data)
        self.assertIsInstance(data, module.ImplicitConstraintDataAbstract)
        self.assertEqual(data.Jc.shape, (3, state.nv))
        self.assertEqual(data.da0_dx.shape, (3, state.ndx))
        self.assertEqual(data.dv0_dq.shape, (3, state.nv))

        with self.assertRaises(crocoddyl.Exception):
            module.ImplicitConstraintDataAbstract(None, pin_data)
        with self.assertRaises(crocoddyl.Exception):
            module.ImplicitConstraintDataAbstract(probe, None)
        with self.assertRaises(crocoddyl.Exception):
            probe.createData(None)
        with self.assertRaises(crocoddyl.Exception):
            module.ImplicitConstraintItem("null", None)

        manager = module.ImplicitConstraintModelMultiple(state)
        manager.addConstraint("probe", probe)
        manager_data = manager.createData(pin_data)
        with self.assertRaises(crocoddyl.Exception):
            module.ImplicitConstraintDataMultiple(None, pin_data)
        with self.assertRaises(crocoddyl.Exception):
            module.ImplicitConstraintDataMultiple(manager, None)
        with self.assertRaises(crocoddyl.Exception):
            manager.createData(None)
        x = np.asarray(state.rand(), dtype=dtype)
        manager.calc(manager_data, x)
        manager.calcDiff(manager_data, x)
        manager.updateForce(manager_data, np.arange(3, dtype=dtype) + dtype(1))
        self.assertEqual(
            (probe.calc_calls, probe.diff_calls, probe.force_calls), (1, 1, 1)
        )
        self.assertTrue(np.all(manager_data.Jc == dtype(2)))
        self.assertTrue(np.all(manager_data.da0_dx == dtype(3)))
        self.assertTrue(np.all(manager_data.dv0_dq == dtype(4)))

        item = manager.constraints["probe"]
        for field, value in (
            ("name", "other"),
            ("constraint", probe),
            ("active", False),
        ):
            with self.assertRaises(AttributeError):
                setattr(item, field, value)
        manager.changeConstraintStatus("probe", False)
        self.assertFalse(item.active)
        self.assertEqual(manager.nc, 0)
        self.assertEqual(set(manager.inactive_set), {"probe"})
        manager.changeConstraintStatus("probe", True)
        self.assertTrue(item.active)
        self.assertEqual(manager.nc, 3)

        reference = state.pinocchio.frames[1].placement
        contact = module.ContactModel(
            state,
            1,
            reference,
            pinocchio.LOCAL,
            state.nv,
            np.zeros(2, dtype=dtype),
            [True, True, True, False, False, False],
        )
        self.assertEqual(contact.nc, 3)
        self.assertEqual(contact.mask, [True, True, True, False, False, False])
        contact_data = contact.createData(pin_data)
        self.assertIsInstance(contact_data, module.ContactData)
        with self.assertRaises(crocoddyl.Exception):
            module.ContactData(None, pin_data)
        with self.assertRaises(crocoddyl.Exception):
            module.ContactData(contact, None)
        with self.assertRaises(crocoddyl.Exception):
            contact.createData(None)
        copied_contact = copy.copy(contact)
        copied_contact_data = copy.copy(contact_data)
        self.assertEqual(copied_contact.nc, contact.nc)
        self.assertEqual(copied_contact.mask, contact.mask)
        self.assertTrue(np.array_equal(copied_contact_data.Jc, contact_data.Jc))
        cast_dtype = (
            crocoddyl.DType.Float32 if module is crocoddyl else crocoddyl.DType.Float64
        )
        self.assertEqual(contact.cast(cast_dtype).nc, 3)
        with self.assertRaises(crocoddyl.Exception):
            module.ContactModel(
                state,
                1,
                reference,
                pinocchio.LOCAL,
                np.zeros(2, dtype=dtype),
                [False] * 6,
            )
        with self.assertRaises(crocoddyl.Exception):
            module.ContactModel(
                state,
                1,
                reference,
                pinocchio.LOCAL,
                np.zeros(2, dtype=dtype),
                [True, False],
            )

        for mask, nc in (
            ([False, False, True, False, False, False], 1),
            ([True, False, True, False, False, False], 2),
            ([True] * 6, 6),
            ([True, False, False, False, True, False], 2),
        ):
            masked = module.ContactModel(
                state,
                1,
                reference,
                pinocchio.LOCAL,
                state.nv,
                np.zeros(2, dtype=dtype),
                mask,
            )
            masked_data = masked.createData(pin_data)
            masked.calc(masked_data, x)
            masked.calcDiff(masked_data, x)
            masked.updateForce(masked_data, np.ones(nc, dtype=dtype))
            self.assertEqual(masked.nc, nc)
            self.assertEqual(np.asarray(masked_data.Jc).size, nc * state.nv)

        contact6 = module.ContactModel(
            state,
            1,
            reference,
            pinocchio.LOCAL,
            state.nv,
            np.zeros(2, dtype=dtype),
            [True] * 6,
        )
        contact_manager = module.ImplicitConstraintModelMultiple(state)
        contact_manager.addConstraint("contact", contact6)
        contact_manager.computeAllConstraints = True
        contact_manager_data = contact_manager.createData(pin_data)
        force_values = np.arange(6, dtype=dtype) + dtype(1)
        force_dfx = np.ones((6, state.ndx), dtype=dtype)
        force_dfu = np.ones((6, state.nv), dtype=dtype)
        contact_manager.updateForce(contact_manager_data, force_values)
        contact_manager.updateForceDiff(contact_manager_data, force_dfx, force_dfu)
        contact_shared = module.DataCollectorImplicitConstraint(contact_manager_data)
        zero_force = type(contact_manager_data.constraints["contact"].f).Zero()
        friction_cone = module.FrictionCone(np.eye(3, dtype=dtype), 0.7)
        wrench_cone = module.WrenchCone(
            np.eye(3, dtype=dtype),
            0.7,
            np.array([0.1, 0.2], dtype=dtype),
        )
        cop_support = module.CoPSupport(
            np.eye(3, dtype=dtype), np.array([0.1, 0.2], dtype=dtype)
        )
        generic_residuals = (
            (
                module.ResidualModelContactForce(state, 1, zero_force, 6),
                force_values,
                force_dfx,
                force_dfu,
            ),
            (
                module.ResidualModelContactFrictionCone(state, 1, friction_cone),
                friction_cone.A @ force_values[:3],
                friction_cone.A @ force_dfx[:3],
                friction_cone.A @ force_dfu[:3],
            ),
            (
                module.ResidualModelContactWrenchCone(state, 1, wrench_cone),
                wrench_cone.A @ force_values,
                wrench_cone.A @ force_dfx,
                wrench_cone.A @ force_dfu,
            ),
            (
                module.ResidualModelContactCoPPosition(state, 1, cop_support),
                cop_support.A @ force_values,
                cop_support.A @ force_dfx,
                cop_support.A @ force_dfu,
            ),
        )
        u = np.linspace(-0.2, 0.2, state.nv, dtype=dtype)
        for residual, expected_r, expected_rx, expected_ru in generic_residuals:
            residual_data = residual.createData(contact_shared)
            residual.calc(residual_data, x, u)
            residual.calcDiff(residual_data, x, u)
            self.assertEqual(residual_data.r.size, residual.nr)
            np.testing.assert_allclose(
                residual_data.r, expected_r, rtol=tolerance, atol=tolerance
            )
            np.testing.assert_allclose(
                residual_data.Rx, expected_rx, rtol=tolerance, atol=tolerance
            )
            np.testing.assert_allclose(
                residual_data.Ru, expected_ru, rtol=tolerance, atol=tolerance
            )

        force_layouts = (
            ([False, False, True, False, False, False], 1),
            ([True, True, True, False, False, False], 3),
            ([True] * 6, 6),
        )
        for mask, nc in force_layouts:
            layout_contact = module.ContactModel(
                state,
                1,
                reference,
                pinocchio.LOCAL,
                state.nv,
                np.zeros(2, dtype=dtype),
                mask,
            )
            layout_manager = module.ImplicitConstraintModelMultiple(state)
            layout_manager.addConstraint("contact", layout_contact)
            layout_data = layout_manager.createData(pin_data)
            values = np.arange(nc, dtype=dtype) + dtype(1)
            dfx = np.arange(nc * state.ndx, dtype=dtype).reshape(nc, state.ndx)
            dfu = np.arange(nc * state.nv, dtype=dtype).reshape(nc, state.nv)
            layout_manager.updateForce(layout_data, values)
            layout_manager.updateForceDiff(layout_data, dfx, dfu)
            layout_shared = module.DataCollectorImplicitConstraint(layout_data)
            residual = module.ResidualModelContactForce(state, 1, zero_force, nc)
            residual_data = residual.createData(layout_shared)
            residual.calc(residual_data, x, u)
            residual.calcDiff(residual_data, x, u)
            self.assertEqual(residual_data.r.size, residual.nr)
            self.assertEqual(residual.nr, nc)
            np.testing.assert_allclose(
                residual_data.r, values, rtol=tolerance, atol=tolerance
            )
            np.testing.assert_allclose(
                np.asarray(residual_data.Rx).reshape(nc, state.ndx),
                dfx,
                rtol=tolerance,
                atol=tolerance,
            )
            np.testing.assert_allclose(
                np.asarray(residual_data.Ru).reshape(nc, state.nv),
                dfu,
                rtol=tolerance,
                atol=tolerance,
            )
            wrong_nc = 5 if nc == 6 else nc + 1
            with self.assertRaises(crocoddyl.Exception):
                module.ResidualModelContactForce(
                    state, 1, zero_force, wrong_nc
                ).createData(layout_shared)

        friction_layouts = (
            ([True, False, True, False, False, False], 2),
            ([True, True, True, False, False, False], 3),
            ([True] * 6, 6),
        )
        for mask, nc in friction_layouts:
            layout_contact = module.ContactModel(
                state,
                1,
                reference,
                pinocchio.LOCAL,
                state.nv,
                np.zeros(2, dtype=dtype),
                mask,
            )
            layout_manager = module.ImplicitConstraintModelMultiple(state)
            layout_manager.addConstraint("contact", layout_contact)
            layout_data = layout_manager.createData(pin_data)
            values = np.arange(nc, dtype=dtype) + dtype(1)
            dfx = np.arange(nc * state.ndx, dtype=dtype).reshape(nc, state.ndx)
            dfu = np.arange(nc * state.nv, dtype=dtype).reshape(nc, state.nv)
            layout_manager.updateForce(layout_data, values)
            layout_manager.updateForceDiff(layout_data, dfx, dfu)
            layout_shared = module.DataCollectorImplicitConstraint(layout_data)
            residual = module.ResidualModelContactFrictionCone(state, 1, friction_cone)
            residual_data = residual.createData(layout_shared)
            residual.calc(residual_data, x, u)
            residual.calcDiff(residual_data, x, u)
            linear_force = np.zeros(3, dtype=dtype)
            linear_dfx = np.zeros((3, state.ndx), dtype=dtype)
            linear_dfu = np.zeros((3, state.nv), dtype=dtype)
            linear_axes = np.flatnonzero(mask[:3])
            linear_force[linear_axes] = values[: len(linear_axes)]
            linear_dfx[linear_axes] = dfx[: len(linear_axes)]
            linear_dfu[linear_axes] = dfu[: len(linear_axes)]
            self.assertEqual(residual_data.r.size, residual.nr)
            np.testing.assert_allclose(
                residual_data.r,
                friction_cone.A @ linear_force,
                rtol=tolerance,
                atol=tolerance,
            )
            np.testing.assert_allclose(
                residual_data.Rx,
                friction_cone.A @ linear_dfx,
                rtol=tolerance,
                atol=tolerance,
            )
            np.testing.assert_allclose(
                residual_data.Ru,
                friction_cone.A @ linear_dfu,
                rtol=tolerance,
                atol=tolerance,
            )

        unsupported_layouts = (
            [True, False, False, False, False, False],
            [True, True, False, False, False, False],
            [True, True, False, True, False, False],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        )
        for mask in unsupported_layouts:
            layout_contact = module.ContactModel(
                state,
                1,
                reference,
                pinocchio.LOCAL,
                state.nv,
                np.zeros(2, dtype=dtype),
                mask,
            )
            layout_manager = module.ImplicitConstraintModelMultiple(state)
            layout_manager.addConstraint("contact", layout_contact)
            layout_shared = module.DataCollectorImplicitConstraint(
                layout_manager.createData(pin_data)
            )
            unsupported_residuals = (
                module.ResidualModelContactForce(state, 1, zero_force, sum(mask)),
                module.ResidualModelContactFrictionCone(state, 1, friction_cone),
                module.ResidualModelContactWrenchCone(state, 1, wrench_cone),
                module.ResidualModelContactCoPPosition(state, 1, cop_support),
            )
            for residual in unsupported_residuals:
                with self.assertRaises(crocoddyl.Exception):
                    residual.createData(layout_shared)

        loop = module.KinematicLoopModel(
            state,
            1,
            state.pinocchio.jointPlacements[1],
            2,
            state.pinocchio.jointPlacements[2],
            pinocchio.LOCAL,
            state.nv,
            np.zeros(2, dtype=dtype),
            [True, False, True, False, True, False],
        )
        self.assertEqual(loop.nc, 3)
        self.assertEqual(loop.mask, [True, False, True, False, True, False])
        loop_data = loop.createData(pin_data)
        self.assertIsInstance(loop_data, module.KinematicLoopData)
        with self.assertRaises(crocoddyl.Exception):
            module.KinematicLoopData(None, pin_data)
        with self.assertRaises(crocoddyl.Exception):
            module.KinematicLoopData(loop, None)
        with self.assertRaises(crocoddyl.Exception):
            loop.createData(None)
        self.assertIsInstance(copy.copy(loop_data), module.KinematicLoopData)
        self.assertEqual(loop.cast(cast_dtype).mask, loop.mask)
        with self.assertRaises(crocoddyl.Exception):
            module.KinematicLoopModel(
                state,
                1,
                state.pinocchio.jointPlacements[1],
                2,
                state.pinocchio.jointPlacements[2],
                pinocchio.LOCAL,
                np.zeros(2, dtype=dtype),
                [False] * 6,
            )
        with self.assertRaises(crocoddyl.Exception):
            module.KinematicLoopModel(
                state,
                1,
                state.pinocchio.jointPlacements[1],
                2,
                state.pinocchio.jointPlacements[2],
                pinocchio.LOCAL,
                np.zeros(2, dtype=dtype),
                [True, False],
            )

        actuation_model = module.ActuationModelMultibody(state)
        actuation = actuation_model.createData()
        actuation_u = np.zeros(actuation_model.nu, dtype=dtype)
        actuation_model.calc(actuation, x, actuation_u)
        actuation_model.calcDiff(actuation, x, actuation_u)
        joint = module.JointDataAbstract(state, actuation_model, actuation_model.nu)
        params = module.ParamsDataAbstract(1, 1)
        collectors = [
            (module.DataCollectorImplicitConstraint(manager_data), ("constraints",)),
            (
                module.DataCollectorMultibodyInImplicitConstraint(
                    pin_data, manager_data
                ),
                ("pinocchio", "constraints"),
            ),
            (
                module.DataCollectorMultibodyInImplicitConstraintParams(
                    pin_data, manager_data, params
                ),
                ("pinocchio", "constraints", "params"),
            ),
            (
                module.DataCollectorActMultibodyInImplicitConstraint(
                    pin_data, actuation, manager_data
                ),
                ("pinocchio", "actuation", "constraints"),
            ),
            (
                module.DataCollectorActMultibodyInImplicitConstraintParams(
                    pin_data, actuation, manager_data, params
                ),
                ("pinocchio", "actuation", "constraints", "params"),
            ),
            (
                module.DataCollectorJointMultibodyInImplicitConstraint(
                    pin_data, joint, manager_data
                ),
                ("pinocchio", "joint", "constraints"),
            ),
            (
                module.DataCollectorJointMultibodyInImplicitConstraintParams(
                    pin_data, joint, manager_data, params
                ),
                ("pinocchio", "joint", "constraints", "params"),
            ),
            (
                module.DataCollectorJointActMultibodyInImplicitConstraint(
                    pin_data, actuation, joint, manager_data
                ),
                ("pinocchio", "actuation", "joint", "constraints"),
            ),
            (
                module.DataCollectorJointActMultibodyInImplicitConstraintParams(
                    pin_data, actuation, joint, manager_data, params
                ),
                ("pinocchio", "actuation", "joint", "constraints", "params"),
            ),
        ]
        for collector, fields in collectors:
            copied = copy.copy(collector)
            for field in fields:
                self.assertTrue(hasattr(collector, field))
                if field != "pinocchio":
                    self.assertIs(getattr(copied, field), getattr(collector, field))

        copied_manager = copy.copy(contact_manager)
        copied_manager_data = copy.copy(contact_manager_data)
        self.assertEqual(copied_manager.nc, contact_manager.nc)
        self.assertEqual(copied_manager.nc_total, contact_manager.nc_total)
        self.assertEqual(copied_manager.nu, contact_manager.nu)
        self.assertEqual(
            copied_manager.computeAllConstraints,
            contact_manager.computeAllConstraints,
        )
        self.assertEqual(
            set(copied_manager.active_set), set(contact_manager.active_set)
        )
        self.assertEqual(
            set(copied_manager.inactive_set), set(contact_manager.inactive_set)
        )
        self.assertIsNot(
            copied_manager.constraints["contact"],
            contact_manager.constraints["contact"],
        )
        self.assertIs(
            copied_manager.constraints["contact"].constraint,
            contact_manager.constraints["contact"].constraint,
        )
        copied_manager.changeConstraintStatus("contact", False)
        self.assertEqual(copied_manager.nc, 0)
        self.assertFalse(copied_manager.constraints["contact"].active)
        self.assertEqual(contact_manager.nc, 6)
        self.assertTrue(contact_manager.constraints["contact"].active)
        copied_manager.changeConstraintStatus("contact", True)
        self.assertEqual(copied_manager.nc, 6)
        self.assertEqual(contact_manager.nc, 6)
        shared_marker = np.full((6, state.nv), dtype(7), dtype=dtype)
        contact_manager_data.constraints["contact"].Jc = shared_marker
        self.assertTrue(
            np.array_equal(copied_manager_data.constraints["contact"].Jc, shared_marker)
        )

        actuation_contact_shared = module.DataCollectorActMultibodyInImplicitConstraint(
            pin_data, actuation, contact_manager_data
        )
        gravity = module.ResidualModelContactControlGrav(state)
        gravity_data = gravity.createData(actuation_contact_shared)
        gravity.calc(gravity_data, x, u)
        gravity.calcDiff(gravity_data, x, u)
        self.assertTrue(np.all(np.isfinite(gravity_data.r)))

        contact_manager.updateVelocity(
            contact_manager_data,
            np.linspace(-0.1, 0.1, state.nv, dtype=dtype),
        )
        contact_manager.updateVelocityDiff(
            contact_manager_data,
            np.eye(state.nv, state.ndx, dtype=dtype),
        )
        multibody_contact_shared = module.DataCollectorMultibodyInImplicitConstraint(
            pin_data, contact_manager_data
        )
        impulse_com = module.ResidualModelImpulseCoM(state)
        impulse_com_data = impulse_com.createData(multibody_contact_shared)
        impulse_com.calc(impulse_com_data, x, np.empty(0, dtype=dtype))
        impulse_com.calcDiff(impulse_com_data, x, np.empty(0, dtype=dtype))
        self.assertEqual(impulse_com_data.r.shape, (3,))

        with self.assertRaises(crocoddyl.Exception):
            manager.addConstraint("null", None)

    def test_float64(self):
        self.check_scalar(crocoddyl, np.float64, self.state64)

    def test_float32(self):
        self.check_scalar(crocoddyl_float32, np.float32, self.state32)


if __name__ == "__main__":
    unittest.main()
