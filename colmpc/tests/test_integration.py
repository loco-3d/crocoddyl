import json
import unittest
from pathlib import Path

import numpy as np
import pinocchio as pin
from ocp import OCPPandaReachingColWithMultipleCol
from scenes import Scene
from wrapper_panda import PandaWrapper, compute_distance_between_shapes


class TestBenchmark(unittest.TestCase):
    @staticmethod
    def load_test_data(test):
        """Load JSON data for the given test."""
        try:
            with (Path("tests_benchmark") / f"{test}.json").open() as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"Test file {test}.json not found.")
            return None
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {test}.json.")
            return None

    @staticmethod
    def extract_collision_pairs(data):
        """Extract and validate collision pairs from data keys."""
        keys = data.keys()
        collision_pairs = []

        for key in keys:
            parts = key.split("-", 1)
            if len(parts) == 2:
                collision_pairs.append((parts[0], parts[1]))
        return collision_pairs

    def test_distances_and_solution(self):
        """
        Test that the distances are well computed within coal and test the solutions
        found by the solver.
        """

        tests = ["box", "wall", "ball"]

        for test in tests:
            print(f"Testing the {test} benchmark.")
            data = self.load_test_data(test)
            if data is None:
                continue

            collision_pairs = self.extract_collision_pairs(data)

            # Creating the scene
            robot_wrapper = PandaWrapper(capsule=False)
            rmodel, cmodel, _vmodel = robot_wrapper()
            scene = Scene()

            cmodel, TARGET, q0 = scene.create_scene(rmodel, cmodel, test)

            ### INITIAL X0
            x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])

            ### CREATING THE PROBLEM WITHOUT WARM START
            problem = OCPPandaReachingColWithMultipleCol(
                rmodel,
                cmodel,
                TARGET,
                T=data["ocp_params"]["T"],
                dt=data["ocp_params"]["dt"],
                x0=x0,
                WEIGHT_GRIPPER_POSE=data["ocp_params"]["WEIGHT_GRIPPER_POSE"],
                WEIGHT_xREG=data["ocp_params"]["WEIGHT_xREG"],
                WEIGHT_uREG=data["ocp_params"]["WEIGHT_uREG"],
                SAFETY_THRESHOLD=data["ocp_params"]["SAFETY_THRESHOLD"],
            )
            ddp = problem()

            XS_init = [x0] * (data["ocp_params"]["T"] + 1)
            # US_init = [np.zeros(rmodel.nv)] * T
            US_init = ddp.problem.quasiStatic(XS_init[:-1])

            # Solving the problem
            ddp.solve(XS_init, US_init)

            np.testing.assert_allclose(
                np.array(ddp.xs.tolist()),
                data["xs"],
                atol=1e-3,
                err_msg="Differences between xs from benchmark and xs computed here.",
            )

            for it, xs in enumerate(ddp.xs):
                for col_pair in collision_pairs:
                    cp1 = col_pair[0]
                    cp2 = col_pair[1]
                    dist = compute_distance_between_shapes(
                        rmodel,
                        cmodel,
                        cmodel.getGeometryId(cp1),
                        cmodel.getGeometryId(cp2),
                        np.array(xs.tolist()[:7]),
                    )
                    dist_pre = data[cp1 + "-" + cp2][it]

                    self.assertAlmostEqual(
                        dist,
                        dist_pre,
                        places=5,
                        msg="Benchmark distances different than the one computed here.",
                    )


if __name__ == "__main__":
    unittest.main()
