import os
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.biped import plotSolution
from crocoddyl.utils.humanoid import HumanoidLocoManipulation

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ

# Load robot
robot = example_robot_data.load("talos")
robot_model = robot.model

# Define t`he initial posture, feet and hands
RF_name, LF_name = "right_sole_link", "left_sole_link"
LH_name, RH_name = "gripper_left_inner_double_link", "gripper_right_inner_double_link"
q0 = robot_model.referenceConfigurations["half_sitting"]

# Define the location of the bars
bars = list()
bars.append(np.array([0.4, 0.0, 0.3]))

# Define the monkey-bar tasks
RH_grasp_list, LH_grasp_list = list(), list()
for bar_pos in bars:
    Rhand = pinocchio.utils.rpyToMatrix(0.0, 0.0, np.pi / 2)
    t_rh = bar_pos - np.array([0.0, 0.45, 0.0])
    t_lh = bar_pos + np.array([0.0, 0.45, 0.0])
    RH_grasp_list.append(pinocchio.SE3(Rhand, t_rh))
    LH_grasp_list.append(pinocchio.SE3(Rhand, t_lh))
foot_distance = bars[-1][0] + 0.7
RF_pose = pinocchio.SE3(np.eye(3), np.array([foot_distance, -0.15, 0.0]))
LF_pose = pinocchio.SE3(np.eye(3), np.array([foot_distance, 0.15, 0.0]))

# Creating the OC problem and solver with its callbacks
humanoid = HumanoidLocoManipulation(
    q0, robot_model, RF_name, LF_name, RH_name, LH_name, fwddyn=False
)
problem = humanoid.createHandstandEquilibriumProblem(
    RH_grasp_list, LH_grasp_list, RF_pose, LF_pose
)
solver = crocoddyl.SolverIntro(problem)
solver.th_minImprove = 1e-1
if WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the OC solver
x0 = np.concatenate([q0, np.zeros(robot_model.nv)])
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
# print("*** SOLVE (FeasShoot) ***")
# solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
# solver.solve(xs, us, 200, False)
print("*** SOLVE (MultiShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve(xs, us, 200, False)
# Ts = int(solver.problem.T / 3)
# print("*** SOLVE (HybridShoot: {Ts}) ***".format_map(locals()))
# solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
# solver.solve(xs, us, 200, False)

# Printing the terminal pose
np.set_printoptions(precision=4, suppress=True)
RF_target = solver.problem.terminalModel.differential.constraints.constraints[
    RF_name + "_pose"
].constraint.residual.reference
LF_target = solver.problem.terminalModel.differential.constraints.constraints[
    LF_name + "_pose"
].constraint.residual.reference
RF_Mterm = pinocchio.SE3ToXYZQUAT(
    solver.problem.terminalData.differential.multibody.pinocchio.oMf[
        robot_model.getFrameId(RF_name)
    ]
)
LF_Mterm = pinocchio.SE3ToXYZQUAT(
    solver.problem.terminalData.differential.multibody.pinocchio.oMf[
        robot_model.getFrameId(LF_name)
    ]
)
print("Target end-effector pose:")
print("   RF position:", RF_target.translation)
print("   RF quaternion:", pinocchio.Quaternion(RF_target.rotation).coeffs())
print("   LF position:", LF_target.translation)
print("   LF quaternion:", pinocchio.Quaternion(LF_target.rotation).coeffs())
print("Terminal end-effector pose:")
print("   RF position:", RF_Mterm[:3])
print("   RF quaternion:", RF_Mterm[3:7])
print("   LF position:", LF_Mterm[:3])
print("   LF quaternion:", LF_Mterm[3:7])

# Display the solution
if WITHDISPLAY:
    # Create the viewer
    display = crocoddyl.MeshcatDisplay(
        robot, frameNames=[LF_name, RF_name, LH_name, RH_name]
    )
    # Display the bars
    import meshcat.geometry as g

    for i, bar_pos in enumerate(bars):
        Mbar = pinocchio.SE3(np.eye(3), bar_pos)
        color = g.MeshLambertMaterial(
            color=display._rgbToHexColor([0.4, 0.4, 0.4, 1.0]),
            reflectivity=0.8,
        )
        display.robot.viewer["bar_{i}".format_map(locals())].set_object(
            g.Cylinder(1.2, 0.035), color
        )
        display.robot.viewer["bar_{i}".format_map(locals())].set_transform(
            Mbar.homogeneous
        )
    # Display the optimized motion
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)

# Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[1]
    plotSolution(solver, bounds=False, figIndex=1, show=False)
    crocoddyl.plotConvergence(
        log.costs, log.pregs, log.dregs, log.grads, log.stops, log.steps, figIndex=3
    )
