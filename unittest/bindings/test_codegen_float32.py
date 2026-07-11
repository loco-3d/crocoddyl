import os
import tempfile
import unittest

import numpy as np
import pinocchio
import pinocchio.cppadcg as pinocg64
import pinocchio.cppadcg_float32 as pinocg32

import crocoddyl
import crocoddyl.cgfloat32 as crocoddylcg32
import crocoddyl.cgfloat64 as crocoddylcg64
import crocoddyl.float32 as crocoddyl32


class CodeGenFloat32Test(unittest.TestCase):
    def test_scalar_modules_and_casts(self):
        self.assertIs(crocoddyl.DType.values[0], crocoddyl.DType.Float64)
        self.assertIs(crocoddyl.DType.values[1], crocoddyl.DType.Float32)
        self.assertIs(crocoddyl.DType.values[2], crocoddyl.DType.ADFloat64)
        self.assertIs(crocoddyl.DType.values[3], crocoddyl.DType.ADFloat32)

        self.assertIsNot(crocoddyl.ActionModelLQR, crocoddyl32.ActionModelLQR)
        self.assertIsNot(crocoddylcg64.ActionModelLQR, crocoddylcg32.ActionModelLQR)
        self.assertIsNot(pinocg64.Model, pinocg32.Model)

        model64 = crocoddyl.ActionModelLQR(4, 2)
        model32 = model64.cast(crocoddyl.DType.Float32)
        model_ad64 = model32.cast(crocoddyl.DType.ADFloat64)
        model_ad32 = model32.cast(crocoddyl.DType.ADFloat32)

        self.assertIsInstance(model32, crocoddyl32.ActionModelLQR)
        self.assertIsInstance(model_ad64, crocoddylcg64.ActionModelLQR)
        self.assertIsInstance(model_ad32, crocoddylcg32.ActionModelLQR)
        self.assertIsInstance(
            model_ad64.cast(crocoddyl.DType.ADFloat32),
            crocoddylcg32.ActionModelLQR,
        )
        self.assertIsInstance(
            model_ad32.cast(crocoddyl.DType.ADFloat64),
            crocoddylcg64.ActionModelLQR,
        )

        state64 = crocoddyl.StateMultibody(pinocchio.buildSampleModelManipulator())
        state32 = state64.cast(crocoddyl.DType.Float32)
        state_ad32 = state32.cast(crocoddyl.DType.ADFloat32)
        self.assertIsInstance(state32, crocoddyl32.StateMultibody)
        self.assertIsInstance(state_ad32, crocoddylcg32.StateMultibody)
        self.assertIsInstance(state_ad32.pinocchio, pinocg32.Model)

    def test_float32_codegen(self):
        model64 = crocoddyl.ActionModelLQR(4, 2)
        model32 = model64.cast(crocoddyl.DType.Float32)
        data32 = model32.createData()

        x = np.linspace(-0.2, 0.2, model32.state.nx, dtype=np.float32)
        u = np.linspace(-0.1, 0.1, model32.nu, dtype=np.float32)
        model32.calc(data32, x, u)
        model32.calcDiff(data32, x, u)

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                generated = crocoddyl32.ActionModelCodeGen(
                    model32, "crocoddyl_lqr_float32"
                )
                generated_data = generated.createData()
                generated.calc(generated_data, x, u)
                generated.calcDiff(generated_data, x, u)

                self.assertEqual(generated_data.xnext.dtype, np.float32)
                for field in (
                    "xnext",
                    "r",
                    "Fx",
                    "Fu",
                    "Lx",
                    "Lu",
                    "Lxx",
                    "Lxu",
                    "Luu",
                ):
                    np.testing.assert_allclose(
                        getattr(generated_data, field),
                        getattr(data32, field),
                        rtol=1e-4,
                        atol=1e-5,
                    )
                self.assertAlmostEqual(generated_data.cost, data32.cost, places=5)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()
