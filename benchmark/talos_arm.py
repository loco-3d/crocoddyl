import sys
sys.path.append("/opt/openrobots/share/example-robot-data/unittest/")
import unittest_utils
import crocoddyl
import pinocchio
import numpy as np

# First, let's load the Pinocchio model for the Talos arm.
ROBOT = unittest_utils.loadTalosArm()
robot_model = ROBOT.model

# Note that we need to include a cost model (i.e. set of cost functions) in
# order to fully define the action model for our optimal control problem.
# For this particular example, we formulate three running-cost functions:
# goal-tracking cost, state and control regularization; and one terminal-cost:
# goal cost. First, let's create the common cost functions.
state = crocoddyl.StateMultibody(robot_model)
Mref = crocoddyl.FramePlacement(robot_model.getFrameId("gripper_left_joint"),
                                pinocchio.SE3(np.eye(3), np.matrix([[.0], [.0], [.4]])))
goalTrackingCost = crocoddyl.CostModelFramePlacement(robot_model, Mref)
xRegCost = crocoddyl.CostModelState(robot_model, state)
uRegCost = crocoddyl.CostModelControl(robot_model)

# Create a cost model per the running and terminal action model.
runningCostModel = crocoddyl.CostModelSum(robot_model)
terminalCostModel = crocoddyl.CostModelSum(robot_model)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-3)
runningCostModel.addCost("xReg", xRegCost, 1e-7)
runningCostModel.addCost("uReg", uRegCost, 1e-7)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 1)

# Next, we need to create an action model for running and terminal knots. The
# forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFullyActuated.
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, runningCostModel), 1e-3)
# runningModel.differential.armature = np.matrix([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0. ]).T
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, terminalCostModel), 1e-3)
# terminalModel.differential.armature = np.matrix([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0. ]).T

# For this optimal control problem, we define 250 knots (or running action
# models) plus a terminal knot
T = 250
q0 = np.matrix([0.173046, 1., -0.52366, 0., 0., 0.1, -0.005]).T
x0 = np.vstack([q0, np.zeros((robot_model.nv, 1))])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverDDP(problem)
# cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
# ddp.callback = [CallbackDDPVerbose()]
# if WITHPLOT:
#     ddp.callback.append(CallbackDDPLogger())
# if WITHDISPLAY:
#     ddp.callback.append(CallbackSolverDisplay(ROBOT, 4, 1, cameraTF))

# Solving it with the DDP algorithm
ddp.solve([], [], 6)

# Plotting the solution and the DDP convergence
# if WITHPLOT:
#     log = ddp.callback[1]
#     plotOCSolution(log.xs, log.us, figIndex=1, show=False)
#     plotDDPConvergence(log.costs, log.control_regs, log.state_regs, log.gm_stops,
#                        log.th_stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
# if WITHDISPLAY:
#     from crocoddyl.diagnostic import displayTrajectory
#     displayTrajectory(ROBOT, ddp.xs, runningModel.timeStep)

# Printing the reached position
frame_idx = robot_model.getFrameId("gripper_left_joint")
print
print("The reached pose by the wrist is")
# print(ddp.datas()[-1].differential.pinocchio.oMf[frame_idx])
