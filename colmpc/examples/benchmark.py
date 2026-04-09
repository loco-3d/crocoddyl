# BSD 3-Clause License
#
# Copyright (C) 2025, LAAS-CNRS.
# Copyright note valid unless otherwise stated in individual files.
# All rights reserved.

import argparse
import time
from pathlib import Path

import create_ocp
import crocoddyl
import pinocchio as pin
from param_parsers import ParamParser
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

# Add argument for scene
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

if args.velocity:
    ocp, objects = create_ocp.create_ocp_velocity(rmodel, cmodel, pp)
else:
    # OCP with distance constraints
    ocp, objects = create_ocp.create_ocp_distance(
        rmodel, cmodel, args.distance_in_cost, pp
    )

frame_placement_residual = objects["framePlacementResidual"]
ref = frame_placement_residual.reference

# import mim_solvers
# ocp.setCallbacks([
#     mim_solvers.CallbackLogger(),
#     mim_solvers.CallbackVerbose(),
# ])

XS_init = [pp.get_X0()] * (pp.get_T() + 1)
US_init = ocp.problem.quasiStatic(XS_init[:-1])

xs = XS_init.copy()
us = US_init.copy()

# crocoddyl.enable_profiler()

for n in range(10):
    start = time.time()
    ocp.solve(xs, us, 50)
    stop = time.time()
    print(f"{n:4}  {(stop - start) * 1e3:6.2f}  {ocp.iter:3}")
    xs = ocp.xs
    us = ocp.us

    ref.translation[0] += 0.01
    frame_placement_residual.reference = ref

crocoddyl.stop_watch_report(2)
