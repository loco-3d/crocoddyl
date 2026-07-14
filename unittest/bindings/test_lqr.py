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


def make_values(dtype, nx=4, nu=3, np_=2, ng=2, nh=2):
    nz = nx + nu + np_
    factor = np.empty((nz, nz), dtype=dtype)
    for i in range(nz):
        for j in range(nz):
            factor[i, j] = dtype(0.01 * (1 + i + 2 * j))
    matrix = factor.T @ factor + dtype(2) * np.eye(nz, dtype=dtype)

    def matrix_values(rows, cols, offset):
        values = np.arange(1, rows * cols + 1, dtype=dtype).reshape(rows, cols)
        return dtype(offset) + dtype(0.03) * values

    def vector_values(size, offset):
        return dtype(offset) + dtype(0.05) * np.arange(1, size + 1, dtype=dtype)

    return {
        "A": matrix_values(nx, nx, 0.11),
        "B": matrix_values(nx, nu, -0.07),
        "P": matrix_values(nx, np_, 0.13),
        "Q": matrix[:nx, :nx],
        "R": matrix[nx : nx + nu, nx : nx + nu],
        "N": matrix[:nx, nx : nx + nu],
        "W": matrix[-np_:, -np_:],
        "Y": matrix[:nx, -np_:],
        "V": matrix[nx : nx + nu, -np_:],
        "G": matrix_values(ng, nz, -0.09),
        "H": matrix_values(nh, nz, 0.17),
        "f": vector_values(nx, 0.2),
        "q": vector_values(nx, -0.3),
        "r": vector_values(nu, 0.4),
        "m": vector_values(np_, -0.5),
        "g": vector_values(ng, 0.6),
        "h": vector_values(nh, -0.7),
    }


class LQRTest(unittest.TestCase):
    def check_constructors_and_legacy_compatibility(self, module, dtype):
        nx, nu, ng, nh = 4, 3, 2, 2
        values = make_values(dtype, nx, nu, 0, ng, nh)
        matrices = module.ActionModelLQR(
            values["A"], values["B"], values["Q"], values["R"], values["N"]
        )
        self.assertEqual(matrices.np, 0)
        self.assertTrue(np.array_equal(matrices.f, np.zeros(nx, dtype=dtype)))

        affine = module.ActionModelLQR(
            values["A"],
            values["B"],
            values["Q"],
            values["R"],
            values["N"],
            values["f"],
            values["q"],
            values["r"],
        )
        self.assertTrue(np.array_equal(affine.f, values["f"]))

        constrained = module.ActionModelLQR(
            values["A"],
            values["B"],
            values["Q"],
            values["R"],
            values["N"],
            values["G"],
            values["H"],
            values["f"],
            values["q"],
            values["r"],
            values["g"],
            values["h"],
        )
        self.assertEqual((constrained.ng, constrained.nh), (ng, nh))

        legacy = module.ActionModelLQR(nx, nu, False)
        parameter_zero = module.ActionModelLQR(nx, nu, 0, 0, 0, False)
        for field in ("A", "B", "P", "Q", "R", "N", "W", "Y", "V", "f", "q", "r"):
            self.assertTrue(
                np.array_equal(getattr(legacy, field), getattr(parameter_zero, field))
            )
        x = np.linspace(-0.4, 0.7, nx, dtype=dtype)
        u = np.linspace(0.2, 0.8, nu, dtype=dtype)
        legacy_data = legacy.createData()
        parameter_data = parameter_zero.createData()
        legacy.calc(legacy_data, x, u)
        parameter_zero.calc(parameter_data, x, u)
        legacy.calcDiff(legacy_data, x, u)
        parameter_zero.calcDiff(parameter_data, x, u)
        self.assertTrue(np.array_equal(legacy_data.xnext, parameter_data.xnext))
        self.assertEqual(legacy_data.cost, parameter_data.cost)
        self.assertTrue(np.array_equal(legacy_data.Lx, parameter_data.Lx))
        self.assertTrue(np.array_equal(legacy_data.Lu, parameter_data.Lu))

        random = module.ActionModelLQR.Random(nx, nu, ng, nh)
        self.assertEqual((random.state.nx, random.nu, random.np), (nx, nu, 0))
        self.assertEqual((random.ng, random.nh), (ng, nh))
        self.assertIsInstance(random.createData(), module.ActionDataLQR)

    def check_parameter_formulas_and_setters(self, module, dtype):
        nx, nu, np_, ng, nh = 4, 3, 2, 2, 2
        values = make_values(dtype, nx, nu, np_, ng, nh)
        model = module.ActionModelLQR(nx, nu, np_, ng, nh, False)
        for field, value in values.items():
            setattr(model, field, value)
            self.assertTrue(np.array_equal(getattr(model, field), value))

        data = model.createData()
        self.assertIsInstance(data, module.ActionDataLQR)
        self.assertEqual(data.Fp.shape, (nx, np_))
        self.assertEqual(data.Lp.shape, (np_,))
        self.assertEqual(data.Lpp.shape, (np_, np_))
        self.assertEqual(data.Lpx.shape, (np_, nx))
        self.assertEqual(data.Lpu.shape, (np_, nu))
        self.assertEqual(data.Gp.shape, (ng, np_))
        self.assertEqual(data.Hp.shape, (nh, np_))

        x = np.linspace(-0.4, 0.8, nx, dtype=dtype)
        u = np.linspace(0.3, 0.9, nu, dtype=dtype)
        p = np.linspace(-0.6, 0.5, np_, dtype=dtype)
        model.update_p(data, p)
        model.calc(data, x, u)
        expected_xnext = (
            values["A"] @ x + values["B"] @ u + values["P"] @ p + values["f"]
        )
        expected_cost = dtype(0.5) * x @ values["Q"] @ x
        expected_cost += dtype(0.5) * u @ values["R"] @ u
        expected_cost += x @ values["N"] @ u
        expected_cost += dtype(0.5) * p @ values["W"] @ p
        expected_cost += x @ values["Y"] @ p
        expected_cost += u @ values["V"] @ p
        expected_cost += values["q"] @ x + values["r"] @ u + values["m"] @ p
        z = np.concatenate((x, u, p))
        self.assertTrue(np.allclose(data.xnext, expected_xnext))
        self.assertTrue(np.allclose(data.cost, expected_cost))
        self.assertTrue(np.allclose(data.g, values["G"] @ z + values["g"]))
        self.assertTrue(np.allclose(data.h, values["H"] @ z + values["h"]))

        model.calcDiff(data, x, u)
        self.assertTrue(np.allclose(data.Fx, values["A"]))
        self.assertTrue(np.allclose(data.Fu, values["B"]))
        self.assertTrue(np.allclose(data.Fp, values["P"]))
        self.assertTrue(
            np.allclose(
                data.Lx,
                values["q"] + values["Q"] @ x + values["N"] @ u + values["Y"] @ p,
            )
        )
        self.assertTrue(
            np.allclose(
                data.Lu,
                values["r"] + values["N"].T @ x + values["R"] @ u + values["V"] @ p,
            )
        )
        self.assertTrue(
            np.allclose(
                data.Lp,
                values["m"] + values["Y"].T @ x + values["V"].T @ u + values["W"] @ p,
            )
        )
        self.assertTrue(np.allclose(data.Lpp, values["W"]))
        self.assertTrue(np.allclose(data.Lpx, values["Y"].T))
        self.assertTrue(np.allclose(data.Lpu, values["V"].T))
        self.assertTrue(np.allclose(data.Gp, values["G"][:, -np_:]))
        self.assertTrue(np.allclose(data.Hp, values["H"][:, -np_:]))

        data.Lu = np.full(nu, 21, dtype=dtype)
        data.Luu = np.full((nu, nu), 22, dtype=dtype)
        data.Lxu = np.full((nx, nu), 23, dtype=dtype)
        data.Lpu = np.full((np_, nu), 24, dtype=dtype)
        model.calc(data, x)
        terminal_cost = dtype(0.5) * x @ values["Q"] @ x
        terminal_cost += dtype(0.5) * p @ values["W"] @ p
        terminal_cost += x @ values["Y"] @ p
        terminal_cost += values["q"] @ x + values["m"] @ p
        terminal_z = np.concatenate((x, np.zeros(nu, dtype=dtype), p))
        self.assertTrue(np.array_equal(data.xnext, x))
        self.assertTrue(np.allclose(data.cost, terminal_cost))
        self.assertTrue(np.allclose(data.g, values["G"] @ terminal_z + values["g"]))
        self.assertTrue(np.allclose(data.h, values["H"] @ terminal_z + values["h"]))
        model.calcDiff(data, x)
        self.assertTrue(
            np.allclose(data.Lx, values["q"] + values["Q"] @ x + values["Y"] @ p)
        )
        self.assertTrue(
            np.allclose(data.Lp, values["m"] + values["Y"].T @ x + values["W"] @ p)
        )
        self.assertTrue(np.array_equal(data.Lu, np.full(nu, 21, dtype=dtype)))
        self.assertTrue(np.array_equal(data.Luu, np.full((nu, nu), 22, dtype=dtype)))
        self.assertTrue(np.array_equal(data.Lxu, np.full((nx, nu), 23, dtype=dtype)))
        self.assertTrue(np.array_equal(data.Lpu, np.full((np_, nu), 24, dtype=dtype)))

        wrong_shapes = {
            "A": (nx + 1, nx),
            "B": (nx, nu + 1),
            "P": (nx, np_ + 1),
            "Q": (nx + 1, nx + 1),
            "R": (nu + 1, nu + 1),
            "N": (nx, nu + 1),
            "W": (np_ + 1, np_ + 1),
            "Y": (nx, np_ + 1),
            "V": (nu, np_ + 1),
            "G": (ng, nx + nu + np_ + 1),
            "H": (nh, nx + nu + np_ + 1),
            "f": (nx + 1,),
            "q": (nx + 1,),
            "r": (nu + 1,),
            "m": (np_ + 1,),
            "g": (ng + 1,),
            "h": (nh + 1,),
        }
        for field, shape in wrong_shapes.items():
            with self.assertRaises(Exception):
                setattr(model, field, np.zeros(shape, dtype=dtype))
        with self.assertRaises(Exception):
            model.update_p(data, np.zeros(np_ + 1, dtype=dtype))

        legacy = make_values(dtype, nx, nu, 0, ng, nh)
        model.setLQR(
            legacy["A"],
            legacy["B"],
            legacy["Q"],
            legacy["R"],
            legacy["N"],
            legacy["G"],
            legacy["H"],
            legacy["f"],
            legacy["q"],
            legacy["r"],
            legacy["g"],
            legacy["h"],
        )
        self.assertTrue(np.array_equal(model.P, np.zeros((nx, np_), dtype=dtype)))
        self.assertTrue(np.array_equal(model.W, np.zeros((np_, np_), dtype=dtype)))
        self.assertTrue(np.array_equal(model.m, np.zeros(np_, dtype=dtype)))

    def check_constant_derivatives_refresh_per_data(self, module, dtype):
        nx, nu, np_, ng, nh = 4, 3, 2, 2, 2
        values = make_values(dtype, nx, nu, np_, ng, nh)
        model = module.ActionModelLQR(nx, nu, np_, ng, nh, False)
        for field, value in values.items():
            setattr(model, field, value)

        first = model.createData()
        second = model.createData()
        x = np.linspace(-0.4, 0.8, nx, dtype=dtype)
        u = np.linspace(0.3, 0.9, nu, dtype=dtype)
        p = np.linspace(-0.6, 0.5, np_, dtype=dtype)
        for data in (first, second):
            model.update_p(data, p)
            model.calc(data, x, u)
            model.calcDiff(data, x, u)

        updated = {
            "A": values["A"] + dtype(0.31),
            "B": values["B"] - dtype(0.27),
            "P": values["P"] + dtype(0.23),
            "Q": dtype(20) * np.eye(nx, dtype=dtype),
            "R": dtype(21) * np.eye(nu, dtype=dtype),
            "N": np.zeros((nx, nu), dtype=dtype),
            "W": dtype(22) * np.eye(np_, dtype=dtype),
            "Y": np.zeros((nx, np_), dtype=dtype),
            "V": np.zeros((nu, np_), dtype=dtype),
            "G": values["G"] + dtype(0.19),
            "H": values["H"] - dtype(0.17),
        }
        for field in ("Q", "R", "W", "A", "B", "P", "N", "Y", "V", "G", "H"):
            setattr(model, field, updated[field])

        first.Fu = np.full((nx, nu), 31, dtype=dtype)
        first.Lu = np.full(nu, 32, dtype=dtype)
        first.Luu = np.full((nu, nu), 33, dtype=dtype)
        first.Lxu = np.full((nx, nu), 34, dtype=dtype)
        first.Lpu = np.full((np_, nu), 35, dtype=dtype)
        first.Gu = np.full((ng, nu), 36, dtype=dtype)
        first.Hu = np.full((nh, nu), 37, dtype=dtype)
        model.calcDiff(first, x)
        self.assertTrue(np.allclose(first.Lxx, updated["Q"]))
        self.assertTrue(np.allclose(first.Lpp, updated["W"]))
        self.assertTrue(np.allclose(first.Lpx, updated["Y"].T))
        self.assertTrue(np.allclose(first.Gx, updated["G"][:, :nx]))
        self.assertTrue(np.allclose(first.Gp, updated["G"][:, -np_:]))
        self.assertTrue(np.allclose(first.Hx, updated["H"][:, :nx]))
        self.assertTrue(np.allclose(first.Hp, updated["H"][:, -np_:]))
        self.assertTrue(np.array_equal(first.Fu, np.full((nx, nu), 31, dtype=dtype)))
        self.assertTrue(np.array_equal(first.Lu, np.full(nu, 32, dtype=dtype)))
        self.assertTrue(np.array_equal(first.Luu, np.full((nu, nu), 33, dtype=dtype)))
        self.assertTrue(np.array_equal(first.Lxu, np.full((nx, nu), 34, dtype=dtype)))
        self.assertTrue(np.array_equal(first.Lpu, np.full((np_, nu), 35, dtype=dtype)))
        self.assertTrue(np.array_equal(first.Gu, np.full((ng, nu), 36, dtype=dtype)))
        self.assertTrue(np.array_equal(first.Hu, np.full((nh, nu), 37, dtype=dtype)))

        model.calcDiff(first, x, u)
        model.calcDiff(second, x, u)
        for data in (first, second):
            self.assertTrue(np.allclose(data.Fx, updated["A"]))
            self.assertTrue(np.allclose(data.Fu, updated["B"]))
            self.assertTrue(np.allclose(data.Fp, updated["P"]))
            self.assertTrue(np.allclose(data.Lxx, updated["Q"]))
            self.assertTrue(np.allclose(data.Lxu, updated["N"]))
            self.assertTrue(np.allclose(data.Luu, updated["R"]))
            self.assertTrue(np.allclose(data.Lpp, updated["W"]))
            self.assertTrue(np.allclose(data.Lpx, updated["Y"].T))
            self.assertTrue(np.allclose(data.Lpu, updated["V"].T))
            self.assertTrue(np.allclose(data.Gx, updated["G"][:, :nx]))
            self.assertTrue(np.allclose(data.Gu, updated["G"][:, nx : nx + nu]))
            self.assertTrue(np.allclose(data.Gp, updated["G"][:, -np_:]))
            self.assertTrue(np.allclose(data.Hx, updated["H"][:, :nx]))
            self.assertTrue(np.allclose(data.Hu, updated["H"][:, nx : nx + nu]))
            self.assertTrue(np.allclose(data.Hp, updated["H"][:, -np_:]))

    def check_manager_copy_and_cast(self, module, dtype, other_module, other_dtype):
        nx, nu, np_, ng, nh = 4, 3, 2, 2, 2
        values = make_values(dtype, nx, nu, np_, ng, nh)
        model = module.ActionModelLQR(nx, nu, np_, ng, nh, False)
        for field, value in values.items():
            setattr(model, field, value)

        params = module.LQRParams(model.state, np_)
        params_from_dimension = module.LQRParams(nx, np_)
        self.assertEqual(
            (params_from_dimension.state.nx, params_from_dimension.np), (nx, np_)
        )
        params.lb = np.full(np_, -2, dtype=dtype)
        params.ub = np.full(np_, 3, dtype=dtype)
        params_data = params.createData()
        p = np.linspace(-0.25, 0.75, np_, dtype=dtype)
        params.update(params_data, p)
        self.assertTrue(np.array_equal(params_data.p, p))
        with self.assertRaises(Exception):
            params.update(params_data, np.zeros(np_ + 1, dtype=dtype))

        manager = module.ParameterManager(model.state)
        manager.addParam("lqr", params)
        manager.addParam("inactive", module.LQRParams(model.state, 1), False)
        self.assertEqual(
            (manager.np, manager.np_action, manager.np_dynamics), (np_, np_, 0)
        )
        manager_data = manager.createData()
        data = model.createData(manager_data)
        self.assertIsInstance(data, module.ActionDataLQR)
        model.set_params(data, manager)
        model.update_p(data, p)
        self.assertTrue(np.array_equal(manager_data.params.p, p))
        self.assertTrue(np.array_equal(manager_data.action_params["lqr"].p, p))
        self.assertTrue(
            np.array_equal(
                manager_data.action_params["inactive"].p, np.zeros(1, dtype=dtype)
            )
        )
        x = np.linspace(-0.4, 0.8, nx, dtype=dtype)
        u = np.linspace(0.3, 0.9, nu, dtype=dtype)
        terminal_values = make_values(dtype, nx, 0, np_, ng, nh)
        terminal = module.ActionModelLQR(nx, 0, np_, ng, nh, False)
        for field, value in terminal_values.items():
            setattr(terminal, field, value)
        terminal_data = terminal.createData(manager_data)
        terminal.set_params(terminal_data, manager)
        manager.update(manager_data, p)
        model.calc(data, x, u)
        model.calcDiff(data, x, u)
        terminal.calc(terminal_data, x)
        terminal.calcDiff(terminal_data, x)
        self.assertTrue(
            np.allclose(
                data.xnext,
                values["A"] @ x + values["B"] @ u + values["P"] @ p + values["f"],
            )
        )
        self.assertTrue(
            np.allclose(
                data.Lp,
                values["m"] + values["Y"].T @ x + values["V"].T @ u + values["W"] @ p,
            )
        )
        self.assertTrue(
            np.allclose(
                terminal_data.Lp,
                terminal_values["m"]
                + terminal_values["Y"].T @ x
                + terminal_values["W"] @ p,
            )
        )
        p_second = np.linspace(0.65, -0.35, np_, dtype=dtype)
        manager.update(manager_data, p_second)
        model.calc(data, x, u)
        model.calcDiff(data, x, u)
        terminal.calc(terminal_data, x)
        terminal.calcDiff(terminal_data, x)
        self.assertTrue(
            np.allclose(
                data.xnext,
                values["A"] @ x
                + values["B"] @ u
                + values["P"] @ p_second
                + values["f"],
            )
        )
        self.assertTrue(
            np.allclose(
                data.Lp,
                values["m"]
                + values["Y"].T @ x
                + values["V"].T @ u
                + values["W"] @ p_second,
            )
        )
        self.assertTrue(
            np.allclose(
                terminal_data.Lp,
                terminal_values["m"]
                + terminal_values["Y"].T @ x
                + terminal_values["W"] @ p_second,
            )
        )
        manager.update(manager_data, p)
        manager.calcDiff_action(manager_data, data, x, u)
        self.assertTrue(
            np.array_equal(manager_data.params.dx_dp, np.zeros((nx, np_), dtype=dtype))
        )

        self.assertIsInstance(model.createData(None), module.ActionDataLQR)
        with self.assertRaises(Exception):
            model.set_params(data, None)
        wrong_manager = module.ParameterManager(model.state)
        wrong_manager.addParam("wrong", module.LQRParams(model.state, np_ + 1))
        with self.assertRaises(Exception):
            model.set_params(data, wrong_manager)
        manager.changeParamStatus("inactive", True)
        self.assertEqual(manager.np, np_ + 1)
        with self.assertRaises(Exception):
            model.set_params(data, manager)

        model_copy = copy.copy(model)
        model_deepcopy = copy.deepcopy(model)
        params_copy = copy.copy(params)
        data_copy = copy.copy(data)
        data_deepcopy = copy.deepcopy(data)
        for copied in (model_copy, model_deepcopy):
            self.assertTrue(np.array_equal(copied.P, values["P"]))
            self.assertTrue(np.array_equal(copied.W, values["W"]))
            self.assertTrue(np.array_equal(copied.m, values["m"]))
        self.assertTrue(np.array_equal(params_copy.lb, params.lb))
        for copied in (data_copy, data_deepcopy):
            self.assertTrue(np.array_equal(copied.Fp, data.Fp))
            self.assertTrue(np.array_equal(copied.Lp, data.Lp))
            self.assertTrue(np.array_equal(copied.Lpp, data.Lpp))
            self.assertTrue(np.array_equal(copied.Lpx, data.Lpx))
            self.assertTrue(np.array_equal(copied.Lpu, data.Lpu))
            self.assertTrue(np.array_equal(copied.Gp, data.Gp))
            self.assertTrue(np.array_equal(copied.Hp, data.Hp))
        data.Fp = np.zeros_like(data.Fp)
        self.assertTrue(np.array_equal(data_copy.Fp, values["P"]))

        target_dtype = (
            crocoddyl.DType.Float32 if dtype is np.float64 else crocoddyl.DType.Float64
        )
        casted_model = model.cast(target_dtype)
        casted_params = params.cast(target_dtype)
        self.assertIsInstance(casted_model, other_module.ActionModelLQR)
        self.assertIsInstance(casted_params, other_module.LQRParams)
        self.assertEqual((casted_model.np, casted_params.np), (np_, np_))
        self.assertTrue(np.allclose(casted_model.P, values["P"].astype(other_dtype)))
        self.assertTrue(np.allclose(casted_model.W, values["W"].astype(other_dtype)))
        self.assertTrue(np.allclose(casted_model.m, values["m"].astype(other_dtype)))
        self.assertTrue(np.array_equal(casted_params.lb, params.lb.astype(other_dtype)))
        self.assertTrue(np.array_equal(casted_params.ub, params.ub.astype(other_dtype)))

    def test_float64(self):
        self.check_constructors_and_legacy_compatibility(crocoddyl, np.float64)
        self.check_parameter_formulas_and_setters(crocoddyl, np.float64)
        self.check_constant_derivatives_refresh_per_data(crocoddyl, np.float64)
        self.check_manager_copy_and_cast(
            crocoddyl,
            np.float64,
            crocoddyl_float32,
            np.float32,
        )

    def test_float32(self):
        self.check_constructors_and_legacy_compatibility(crocoddyl_float32, np.float32)
        self.check_parameter_formulas_and_setters(crocoddyl_float32, np.float32)
        self.check_constant_derivatives_refresh_per_data(crocoddyl_float32, np.float32)
        self.check_manager_copy_and_cast(
            crocoddyl_float32,
            np.float32,
            crocoddyl,
            np.float64,
        )


if __name__ == "__main__":
    unittest.main()
