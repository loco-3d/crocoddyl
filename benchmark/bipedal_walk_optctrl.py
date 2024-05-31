import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.biped import SimpleBipedGaitProblem

T = int(sys.argv[1]) if (len(sys.argv) > 1) else int(5e3)  # number of trials
MAXITER = 1
GAIT = "walking"  # 55 nodes


def createProblem(gait_phase):
    robot_model = example_robot_data.load("talos_legs").model
    rightFoot, leftFoot = "right_sole_link", "left_sole_link"
    gait = SimpleBipedGaitProblem(robot_model, rightFoot, leftFoot)
    q0 = robot_model.referenceConfigurations["half_sitting"].copy()
    v0 = pinocchio.utils.zero(robot_model.nv)
    x0 = np.concatenate([q0, v0])

    type_of_gait = next(iter(gait_phase.keys()))
    value = gait_phase[type_of_gait]
    if type_of_gait == "walking":
        # Creating a walking problem
        problem = gait.createWalkingProblem(
            x0,
            value["stepLength"],
            value["stepHeight"],
            value["timeStep"],
            value["stepKnots"],
            value["supportKnots"],
        )

    xs = [robot_model.defaultState] * (len(problem.runningModels) + 1)
    us = [
        m.quasiStatic(d, robot_model.defaultState)
        for m, d in list(zip(problem.runningModels, problem.runningDatas))
    ]
    return xs, us, problem


def runDDPSolveBenchmark(xs, us, problem):
    solver = crocoddyl.SolverFDDP(problem)

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


# Setting up all tasks
if GAIT == "walking":
    GAITPHASE = {
        "walking": {
            "stepLength": 0.6,
            "stepHeight": 0.1,
            "timeStep": 0.0375,
            "stepKnots": 25,
            "supportKnots": 1,
        }
    }

xs, us, problem = createProblem(GAITPHASE)
print("NQ:", problem.terminalModel.state.nq)
print("Number of nodes:", problem.T)
avrg_dur, min_dur, max_dur = runDDPSolveBenchmark(xs, us, problem)
print(f"  FDDP.solve [ms]: {avrg_dur:.4f} ({min_dur:.4f}-{max_dur:.4f})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcBenchmark(xs, us, problem)
print(f"  ShootingProblem.calc [ms]: {avrg_dur:.4f} ({min_dur:.4f}-{max_dur:.4f})")
avrg_dur, min_dur, max_dur = runShootingProblemCalcDiffBenchmark(xs, us, problem)
print(f"  ShootingProblem.calcDiff [ms]: {avrg_dur:.4f} ({min_dur:.4f}-{max_dur:.4f})")
