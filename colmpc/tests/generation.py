import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pinocchio as pin
from ocp import OCPPandaReachingColWithMultipleCol
from scenes import Scene
from wrapper_panda import PandaWrapper, compute_distance_between_shapes

### PARAMETERS
T = 20  # Number of nodes of the trajectory
dt = 0.01  # Time step between each node


# Creating the robot
robot_wrapper = PandaWrapper(capsule=False)
rmodel, cmodel, vmodel = robot_wrapper()

# Creating the scene
scene = Scene()
name_scene = "box"
cmodel, TARGET, q0 = scene.create_scene(rmodel, cmodel, name_scene)

### INITIAL X0
x0 = np.concatenate([q0, pin.utils.zero(rmodel.nv)])

### CREATING THE PROBLEM WITHOUT WARM START
problem = OCPPandaReachingColWithMultipleCol(
    rmodel,
    cmodel,
    TARGET,
    T,
    dt,
    x0,
    WEIGHT_GRIPPER_POSE=100,
    WEIGHT_xREG=1e-2,
    WEIGHT_uREG=1e-4,
    SAFETY_THRESHOLD=2.5e-3,
)
ddp = problem()

XS_init = [x0] * (T + 1)
US_init = ddp.problem.quasiStatic(XS_init[:-1])

# Solving the problem
ddp.solve(XS_init, US_init)
result = defaultdict(list)
for xs in ddp.xs:
    result["xs"].append(xs.tolist())
    for collision_pair in cmodel.collisionPairs:
        id1 = collision_pair.first
        id2 = collision_pair.second
        name_collision = (
            cmodel.geometryObjects[id1].name + "-" + cmodel.geometryObjects[id2].name
        )
        dist = compute_distance_between_shapes(
            rmodel, cmodel, id1, id2, np.array(xs.tolist()[:7])
        )
        result[name_collision].append(dist)

result["T"] = T
result["dt"] = dt
result["ocp_params"] = {
    "T": T,
    "dt": dt,
    "WEIGHT_GRIPPER_POSE": 100,
    "WEIGHT_xREG": 1e-2,
    "WEIGHT_uREG": 1e-4,
    "SAFETY_THRESHOLD": 2.5e-3,
}
with (Path("tests_benchmark") / f"{name_scene}.json").open("w") as outfile:
    json.dump(result, outfile)
