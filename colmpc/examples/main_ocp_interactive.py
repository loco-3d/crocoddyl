# BSD 3-Clause License
#
# Copyright (C) 2025, LAAS-CNRS.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.

import argparse
from pathlib import Path

import create_ocp
import pinocchio as pin
import simulation
from param_parsers import ParamParser
from visualizer import create_viewer
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

parser.add_argument(
    "--distance-in-cost",
    "-d",
    action="store_true",
    help="Add the distance residual to the cost.",
)

parser.add_argument(
    "--velocity",
    "-v",
    action="store_true",
    help="Use velocity-based constraints.",
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
goal_frame_id = rmodel.addFrame(
    pin.Frame("goal", 0, 0, pp.get_target_pose(), pin.FrameType.OP_FRAME)
)
vis = create_viewer(rmodel, cmodel, cmodel)
vis.displayFrames(True, [rmodel.getFrameId("panda2_hand_tcp"), goal_frame_id])
# add_sphere_to_viewer(
# vis, "goal", 5e-2, pp.get_target_pose().translation, color=0x006400
# )

if args.velocity:
    ocp, objects = create_ocp.create_ocp_velocity(rmodel, cmodel, pp)
else:
    # OCP with distance constraints
    ocp, objects = create_ocp.create_ocp_distance_2(
        rmodel, cmodel, args.distance_in_cost, pp
    )

simulation.simulation_loop(
    ocp,
    rmodel,
    goal_frame_id,
    cmodel,
    cmodel.ngeoms - 1,
    objects["framePlacementResidual"],
    pp,
    vis,
)
