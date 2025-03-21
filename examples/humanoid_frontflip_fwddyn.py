import copy
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
q0 = copy.deepcopy(robot_model.referenceConfigurations["half_sitting"])

# Creating the OC problem and solver with its callbacks
humanoid = HumanoidLocoManipulation(q0, robot_model, RF_name, LF_name, RH_name, LH_name)
problem = humanoid.createFlipProblem(0.3)
solver = crocoddyl.SolverFDDP(problem)
solver.th_minImprove = 1e-1
if WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the OC solver
xs, R = [], np.eye(3)
xs.append(
    np.concatenate(
        [
            pinocchio.SE3ToXYZQUAT(pinocchio.SE3(R, np.zeros(3))),
            q0[7:],
            np.zeros(robot_model.nv),
        ]
    )
)
for i in range(solver.problem.T):
    R = R @ pinocchio.utils.rpyToMatrix(0, 2 * np.pi / solver.problem.T, 0)
    xs.append(
        np.concatenate(
            [
                pinocchio.SE3ToXYZQUAT(pinocchio.SE3(R, np.zeros(3))),
                q0[7:],
                np.zeros(robot_model.nv),
            ]
        )
    )
us = solver.problem.quasiStatic(xs[:-1])

# print("*** SOLVE (FeasShoot) ***")
# solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
# solver.solve(xs, us, 300, False)
print("*** SOLVE (MultiShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve(xs, us, 300, False)
# Ts = int(solver.problem.T / 3)
# print("*** SOLVE (HybridShoot: {Ts}) ***".format_map(locals()))
# solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
# solver.solve(xs, us, 300, False)

# Display the solution
if WITHDISPLAY:
    # Create the viewer
    try:
        import gepetto

        gepetto.corbaserver.Client()
        cameraTF = [3.0, 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
        display = crocoddyl.GepettoDisplay(robot, 4, 4, cameraTF)
    except Exception:
        display = crocoddyl.MeshcatDisplay(robot)
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
