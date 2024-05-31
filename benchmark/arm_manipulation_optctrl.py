import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl

# First, let's load the Pinocchio model for the Talos arm.
ROBOT = example_robot_data.load("kinova")
N = 100  # number of nodes
T = int(sys.argv[1]) if (len(sys.argv) > 1) else int(5e3)  # number of trials
MAXITER = 1
CALLBACKS = False


def createProblem(model):
    robot_model = ROBOT.model
    q0 = robot_model.referenceConfigurations["arm_up"]
    x0 = np.concatenate([q0, np.zeros(robot_model.nv)])

    # Note that we need to include a cost model (i.e. set of cost functions) in
    # order to fully define the action model for our optimal control problem.
    # For this particular example, we formulate three running-cost functions:
    # goal-tracking cost, state and control regularization; and one terminal-cost:
    # goal cost. First, let's create the common cost functions.
    state = crocoddyl.StateMultibody(robot_model)
    goalTrackingCost = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ResidualModelFramePlacement(
            state,
            robot_model.getFrameId("j2s6s200_end_effector"),
            pinocchio.SE3(np.eye(3), np.array([0.6, 0.2, 0.5])),
        ),
    )
    xRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelState(state))
    uRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state))

    # Create a cost model per the running and terminal action model.
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)

    # Then let's added the running and terminal cost functions
    runningCostModel.addCost("gripperPose", goalTrackingCost, 1)
    runningCostModel.addCost("xReg", xRegCost, 1e-1)
    runningCostModel.addCost("uReg", uRegCost, 1e-1)
    terminalCostModel.addCost("gripperPose", goalTrackingCost, 1e3)

    # Next, we need to create an action model for running and terminal knots. The
    # forward dynamics (computed using ABA) are implemented
    # inside DifferentialActionModelFullyActuated.
    actuation = crocoddyl.ActuationModelFull(state)
    runningModel = crocoddyl.IntegratedActionModelEuler(
        model(state, actuation, runningCostModel), 1e-2
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        model(state, actuation, terminalCostModel), 0.0
    )

    # For this optimal control problem, we define 100 knots (or running action
    # models) plus a terminal knot
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * N, terminalModel)
    xs = [x0] * (len(problem.runningModels) + 1)
    us = [
        m.quasiStatic(d, x0)
        for m, d in list(zip(problem.runningModels, problem.runningDatas))
    ]
    return xs, us, problem


def runDDPSolveBenchmark(xs, us, problem):
    solver = crocoddyl.SolverFDDP(problem)
    if CALLBACKS:
        solver.setCallbacks([crocoddyl.CallbackVerbose()])
    duration = []
    for _ in range(T):
        c_start = time.time()
        solver.solve(xs, us, MAXITER, False, 0.1)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_duration = sum(duration) / len(duration)
    min_duration = min(duration)
    max_duration = max(duration)
    return avrg_duration, min_duration, max_duration


def runShootingProblemCalcBenchmark(xs, us, problem):
    duration = []
    for _ in range(T):
        c_start = time.time()
        problem.calc(xs, us)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_duration = sum(duration) / len(duration)
    min_duration = min(duration)
    max_duration = max(duration)
    return avrg_duration, min_duration, max_duration


def runShootingProblemCalcDiffBenchmark(xs, us, problem):
    duration = []
    for _ in range(T):
        c_start = time.time()
        problem.calcDiff(xs, us)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_duration = sum(duration) / len(duration)
    min_duration = min(duration)
    max_duration = max(duration)
    return avrg_duration, min_duration, max_duration


np.set_printoptions(precision=2)
xs, us, problem = createProblem(crocoddyl.DifferentialActionModelFreeFwdDynamics)
print("NQ:", problem.terminalModel.state.nq)
print("Number of nodes:", problem.T)
avrg_dur, min_dur, max_dur = runDDPSolveBenchmark(xs, us, problem)
print(f"  FDDP.solve [ms]: {avrg_dur:.4f} ({min_dur:.4f}-{max_dur:.4f})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcBenchmark(xs, us, problem)
print(f"  ShootingProblem.calc [ms]: {avrg_dur:.4f} ({min_dur:.4f}-{max_dur:.4f})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcDiffBenchmark(xs, us, problem)
print(f"  ShootingProblem.calcDiff [ms]: {avrg_dur:.4f} ({min_dur:.4f}-{max_dur:.4f})")
