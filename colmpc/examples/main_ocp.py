# BSD 3-Clause License
#
# Copyright (C) 2024, LAAS-CNRS.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.

import argparse
from pathlib import Path

import create_ocp
import numpy as np
import pinocchio as pin
from param_parsers import ParamParser
from visualizer import add_cube_to_viewer, add_sphere_to_viewer, create_viewer
from wrapper_panda import PandaWrapper


### Argument parser
def scene_type(value):
    ivalue = int(value)
    if ivalue < 1 or ivalue > 3:
        raise argparse.ArgumentTypeError(
            f"Scene must be an integer between 1 and 3. You provided {ivalue}."
        )
    return ivalue


# Create the parser
parser = argparse.ArgumentParser(description="Process scene argument.")

# Add argument for scene
parser.add_argument(
    "--scene",
    "-s",
    type=scene_type,
    required=True,
    help="Scene number (must be 1, 2, or 3).",
)

# Add argument to add distance residual in cost
parser.add_argument(
    "--distance-in-cost",
    "-d",
    action="store_true",
    help="Add the distance residual to the cost.",
)

# Parse the arguments
args = parser.parse_args()


# Creating the robot
robot_wrapper = PandaWrapper(capsule=False)
rmodel, cmodel, vmodel = robot_wrapper()

yaml_path = Path(__file__).parent / "scenes.yaml"
pp = ParamParser(yaml_path, args.scene)

cmodel = pp.add_collisions(rmodel, cmodel)

cdata = cmodel.createData()
rdata = rmodel.createData()

# Generating the meshcat visualizer
vis = create_viewer(rmodel, cmodel, cmodel)
add_sphere_to_viewer(
    vis, "goal", 5e-2, pp.get_target_pose().translation, color=0x006400
)

# OCP with distance constraints
OCP_dist, _ = create_ocp.create_ocp_distance(rmodel, cmodel, args.distance_in_cost, pp)
XS_init = [pp.get_X0()] * (pp.get_T() + 1)
US_init = OCP_dist.problem.quasiStatic(XS_init[:-1])

OCP_dist.solve(XS_init, US_init, 100)
print("OCP with distance constraints solved")

# OCP with velocity constraints
ocp_vel, _ = create_ocp.create_ocp_velocity(rmodel, cmodel, pp)
ocp_vel.solve(XS_init, US_init, 100)


print("OCP with velocity constraints solved")
for i, xs in enumerate(ocp_vel.xs):
    q = np.array(xs[:7].tolist())
    pin.framesForwardKinematics(rmodel, rdata, q)
    add_cube_to_viewer(
        vis,
        f"vcolmpc{i}",
        [2e-2, 2e-2, 2e-2],
        rdata.oMf[rmodel.getFrameId("panda2_rightfinger")].translation,
        color=100000000,
    )

for i, xs in enumerate(OCP_dist.xs):
    q = np.array(xs[:7].tolist())
    pin.framesForwardKinematics(rmodel, rdata, q)
    add_sphere_to_viewer(
        vis,
        f"colmpc{i}",
        2e-2,
        rdata.oMf[rmodel.getFrameId("panda2_rightfinger")].translation,
        color=100000,
    )
print("Press 'ENTER' to display the solution")
while True:
    print("Trajectory of the OCP with distance constraints")
    for k, q in enumerate(OCP_dist.xs):
        vis.display(q[:7])
        input(k)
    print("Trajectory of the OCP with velocity constraints")
    for k, q in enumerate(ocp_vel.xs):
        vis.display(q[:7])
        input(k)
