import unittest

import crocoddyl
import numdifftools as nd
import numpy as np
import pinocchio as pin
from compute_deriv import (
    compute_d_d_dot_dq_dq_dot,
    compute_d_dist_dq,
    compute_ddot,
    compute_dist,
)
from scenes import Scene
from wrapper_panda import PandaWrapper

from colmpc import ResidualModelVelocityAvoidance

np.set_printoptions(precision=7, linewidth=350, suppress=True, threshold=1e6)


def numdiff(f, q, h=1e-6):
    # print(f"numdiff")
    # print(f"q: {q[:7]}")
    # print(f"v: {q[7:]}")
    # print("---------")

    j_diff = np.zeros(len(q))
    fx = f(q)
    for i in range(len(q)):
        e = np.zeros(len(q))
        e[i] = h
        j_diff[i] = (f(q + e) - fx) / e[i]
    return j_diff


class TestResidualModelVelocityAvoidance(crocoddyl.ResidualModelAbstract):
    """
    Class computing the residual of the collision constraint. This residual is simply
    the signed distance between the two closest points of the 2 shapes.
    """

    def __init__(self, state, geom_model: pin.Model, pair_id, ksi=1, di=5e-2, ds=1e-4):
        """
        Class computing the residual of the collision constraint. This residual is
        simply the signed distance between the two closest points of the 2 shapes.

        Args:
            state (crocoddyl.StateMultibody): _description_
            geom_model (pin.Model): Collision model of pinocchio
            pair_id (int): ID of the collision pair
        """
        crocoddyl.ResidualModelAbstract.__init__(self, state, 1, True, True, True)

        # Pinocchio robot model
        self._pinocchio = self.state.pinocchio

        # Geometry model of the robot
        self._geom_model = geom_model

        cp = self._geom_model.collisionPairs[pair_id]
        self.idg1 = cp.first
        self.idg2 = cp.second

        self.ksi = ksi
        self.di = di
        self.ds = ds

    def f(self, x, u=None):
        d_dot = compute_ddot(
            self._pinocchio, self._geom_model, x[:7], x[7:], self.idg1, self.idg2
        )
        d = compute_dist(
            self._pinocchio, self._geom_model, x[:7], x[7:], self.idg1, self.idg2
        )

        return d_dot + self.ksi * (d - self.ds) / (self.di - self.ds)

    def calc(self, data, x, u=None):
        return self.f(x, u)

    def calcDiff(self, data, x, u=None):
        ddistdot_dq, ddistdot_dq_dot = compute_d_d_dot_dq_dq_dot(
            self._pinocchio, self._geom_model, x[:7], x[7:], self.idg1, self.idg2
        )

        # ddistdot_dq_nd = numdiff(
        #     lambda var: compute_ddot(self._pinocchio, self._geom_model, var[:7],
        #     var[7:], self.idg1,  self.idg2), x)
        # print(f"ddistdot_dq_val_test - ddistdot_dq_val_nd : {ddistdot_dq -
        # ddistdot_dq_nd[:7]}")
        # print(f"x: {x}")

        # assert np.allclose(ddistdot_dq_val_nd[:7], ddistdot_dq_val_test, atol=1e-3),
        # f"ddistdot_dq_val_nd: {ddistdot_dq_val_nd[:7]} != {ddistdot_dq_val_test},
        # {ddistdot_dq_val_nd[:7] - ddistdot_dq_val_test}, x: {x.tolist()}"

        ddist_dq = np.r_[
            compute_d_dist_dq(
                self._pinocchio, self._geom_model, x[:7], x[7:], self.idg1, self.idg2
            ),
            np.zeros(self._pinocchio.nq),
        ]
        # ddist_dq_nd = numdiff(lambda var: compute_dist(self._pinocchio,
        # self._geom_model, var[:7], var[7:], self.idg1, self.idg2), x)

        # assert np.allclose(ddist_dq[:7], ddist_dq_nd[:7], atol=1e-3), f"ddist_dq:
        # {ddist_dq[:7]} != {ddist_dq_nd[:7]}, {ddist_dq[:7] - ddist_dq_nd[:7]}, x:
        # {x.tolist()}"

        return np.r_[ddistdot_dq, ddistdot_dq_dot] - ddist_dq * self.ksi / (
            self.di - self.ds
        )
        # data.Rx[:] = ndf


class TestVelocityAvoidance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create random number generator
        cls.rng = np.random.default_rng(2137)

        # Load robot
        robot_wrapper = PandaWrapper(capsule=False)
        cls.rmodel, cmodel, _ = robot_wrapper()
        cls.rdata = cls.rmodel.createData()

        scene = Scene()
        # Update collision model
        cls.cmodel = scene.create_scene(cmodel, scene=4)
        cls.cdata = cls.cmodel.createData()

        # Create state and action model
        cls.state = crocoddyl.StateMultibody(cls.rmodel)
        actuation = crocoddyl.ActuationModelFull(cls.state)
        actuation_data = actuation.createData()
        shared_data = crocoddyl.DataCollectorActMultibody(cls.rdata, actuation_data)

        # Initialize C++ implementation
        cls.velocity_avoidance_residual = ResidualModelVelocityAvoidance(
            cls.state, cls.cmodel, 0, ksi=1, di=5e-2, ds=1e-4
        )
        cls.residual_data = cls.velocity_avoidance_residual.createData(shared_data)

        # Initialize Python test implementation
        cls.test_residual = TestResidualModelVelocityAvoidance(
            cls.state, cls.cmodel, 0, ksi=1, di=5e-2, ds=1e-4
        )

    def _update_placement(self, q: np.array, v: np.array) -> None:
        pin.forwardKinematics(self.state.pinocchio, self.residual_data.pinocchio, q, v)
        pin.updateGeometryPlacements(
            self.state.pinocchio,
            self.residual_data.pinocchio,
            self.cmodel,
            self.cdata,
        )

    def test_calc(self):
        for _ in range(200):
            # Generate new random configuration
            q = pin.randomConfiguration(self.rmodel)
            v = self.rng.random(q.shape)

            # Update poses and compute new residual value
            self._update_placement(q, v)
            # Create state vector
            x = np.concatenate((q, v))

            self.velocity_avoidance_residual.calc(
                self.residual_data, x, np.zeros(q.shape)
            )

            # Compute value of reference residual
            test_residual_result = self.test_residual.calc(
                self.residual_data, x, np.zeros(q.shape)
            )

            self.assertAlmostEqual(
                self.residual_data.r[0],
                test_residual_result,
                places=2,
                msg=(
                    "Result missmatch in function ``calc`` between Python and C++"
                    "implementation!."
                ),
            )

    @unittest.skip("Numdiff is not working")
    def test_calc_diff_finite(self):
        def calc_wrapper(x: np.array) -> float:
            self._update_placement(x[: self.rmodel.nq], x[self.rmodel.nq :])
            self.velocity_avoidance_residual.calc(
                self.residual_data, x, np.zeros(len(x) // 2)
            )
            return self.residual_data.r[0]

        calcDiff_numerical = nd.Gradient(calc_wrapper)

        for _ in range(200):
            # Generate new random configuration
            q = pin.randomConfiguration(self.rmodel)
            v = self.rng.random(q.shape)

            # Update poses and compute new residual value
            self._update_placement(q, v)
            # Create state vector
            x = np.concatenate((q, v))

            # Compute exact derivative
            self.velocity_avoidance_residual.calc(
                self.residual_data, x, np.zeros(q.shape)
            )
            self.velocity_avoidance_residual.calcDiff(
                self.residual_data, x, np.zeros(q.shape)
            )

            Rx_py = self.test_residual.calcDiff(
                self.residual_data, x, np.zeros(q.shape)
            )

            np.testing.assert_allclose(
                self.residual_data.Rx,
                Rx_py,
                atol=1e-2,
                err_msg="Result missmatch in function ``calcDiff`` "
                "between C++ implementation and python implementation!.",
            )

            # Compute approximate derivative
            Rx_num = calcDiff_numerical(x)

            np.testing.assert_allclose(
                self.residual_data.Rx,
                Rx_num,
                rtol=1e-5,
                err_msg="Result missmatch in function ``calcDiff`` "
                "between C++ implementation and finite differences!.",
            )


if __name__ == "__main__":
    unittest.main()
