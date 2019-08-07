import crocoddyl
import pinocchio
# import utils
import numpy as np
import time
import sys
sys.path.append("/opt/openrobots/share/example-robot-data/unittest/")
import unittest_utils

# First, let's load the Pinocchio model for the Talos arm.
ROBOT = unittest_utils.loadTalosArm()
N = 100  # number of nodes
T = int(5e3)  # number of trials
MAXITER = 1


def runBenchmark(model):
    robot_model = ROBOT.model
    q0 = np.matrix([0.173046, 1., -0.52366, 0., 0., 0.1, -0.005]).T
    x0 = np.vstack([q0, np.zeros((robot_model.nv, 1))])

    # Note that we need to include a cost model (i.e. set of cost functions) in
    # order to fully define the action model for our optimal control problem.
    # For this particular example, we formulate three running-cost functions:
    # goal-tracking cost, state and control regularization; and one terminal-cost:
    # goal cost. First, let's create the common cost functions.
    state = crocoddyl.StateMultibody(robot_model)
    Mref = crocoddyl.FramePlacement(robot_model.getFrameId("gripper_left_joint"),
                                    pinocchio.SE3(np.eye(3), np.matrix([[.0], [.0], [.4]])))
    goalTrackingCost = crocoddyl.CostModelFramePlacement(state, Mref)
    xRegCost = crocoddyl.CostModelState(state)
    uRegCost = crocoddyl.CostModelControl(state)

    # Create a cost model per the running and terminal action model.
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)

    # Then let's added the running and terminal cost functions
    runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-3)
    runningCostModel.addCost("xReg", xRegCost, 1e-7)
    runningCostModel.addCost("uReg", uRegCost, 1e-7)
    terminalCostModel.addCost("gripperPose", goalTrackingCost, 1)

    # Next, we need to create an action model for running and terminal knots. The
    # forward dynamics (computed using ABA) are implemented
    # inside DifferentialActionModelFullyActuated.
    runningModel = crocoddyl.IntegratedActionModelEuler(model(state, runningCostModel), 1e-3)
    runningModel.differential.armature = np.matrix([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.]).T
    terminalModel = crocoddyl.IntegratedActionModelEuler(model(state, terminalCostModel), 1e-3)
    terminalModel.differential.armature = np.matrix([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.]).T

    # For this optimal control problem, we define 100 knots (or running action
    # models) plus a terminal knot
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * N, terminalModel)

    # Creating the DDP solver for this OC problem, defining a logger
    ddp = crocoddyl.SolverDDP(problem)

    duration = []
    for i in range(T):
        c_start = time.time()
        ddp.solve([], [], MAXITER)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_duration = sum(duration) / len(duration)
    min_duration = min(duration)
    max_duration = max(duration)
    return avrg_duration, min_duration, max_duration


print('cpp-wrapped free-forward dynamics:')
avrg_duration, min_duration, max_duration = runBenchmark(crocoddyl.DifferentialActionModelFreeFwdDynamics)
print('  CPU time [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))

# TODO @Carlos this is not possible without making an abstract of createData.
# At the time being, I don't have a solution
# print('Python-derived free-forward dynamics:')
# avrg_duration, min_duration, max_duration, tm = runBenchmark(utils.DifferentialFreeFwdDynamicsDerived)
# print('  CPU time [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
