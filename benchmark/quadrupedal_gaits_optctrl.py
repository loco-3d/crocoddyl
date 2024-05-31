import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem

T = int(sys.argv[1]) if (len(sys.argv) > 1) else int(5e3)  # number of trials
MAXITER = 1
CALLBACKS = False
WALKING = "walk" in sys.argv
TROTTING = "trot" in sys.argv
PACING = "pace" in sys.argv
BOUNDING = "bound" in sys.argv
JUMPING = "jump" in sys.argv

GAIT = "walking"  # 104 nodes
if WALKING:
    print("running walking benchmark ...")
    GAIT = "walking"  # 104 nodes
if TROTTING:
    print("running trotting benchmark ...")
    GAIT = "trotting"  # 54 nodes
if PACING:
    print("running pacing benchmark ...")
    GAIT = "pacing"  # 54 nodes
if BOUNDING:
    print("running bounding benchmark ...")
    GAIT = "bounding"  # 54 nodes
if JUMPING:
    print("running jumping benchmark ...")
    GAIT = "jumping"  # 61 nodes


def createProblem(gait_phase):
    robot_model = example_robot_data.load("hyq").model
    lfFoot, rfFoot, lhFoot, rhFoot = "lf_foot", "rf_foot", "lh_foot", "rh_foot"
    gait = SimpleQuadrupedalGaitProblem(robot_model, lfFoot, rfFoot, lhFoot, rhFoot)
    q0 = robot_model.referenceConfigurations["standing"].copy()
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
    elif type_of_gait == "trotting":
        # Creating a trotting problem
        problem = gait.createTrottingProblem(
            x0,
            value["stepLength"],
            value["stepHeight"],
            value["timeStep"],
            value["stepKnots"],
            value["supportKnots"],
        )
    elif type_of_gait == "pacing":
        # Creating a pacing problem
        problem = gait.createPacingProblem(
            x0,
            value["stepLength"],
            value["stepHeight"],
            value["timeStep"],
            value["stepKnots"],
            value["supportKnots"],
        )
    elif type_of_gait == "bounding":
        # Creating a bounding problem
        problem = gait.createBoundingProblem(
            x0,
            value["stepLength"],
            value["stepHeight"],
            value["timeStep"],
            value["stepKnots"],
            value["supportKnots"],
        )
    elif type_of_gait == "jumping":
        # Creating a jumping problem
        problem = gait.createJumpingProblem(
            x0,
            value["jumpHeight"],
            value["jumpLength"],
            value["timeStep"],
            value["groundKnots"],
            value["flyingKnots"],
        )

    xs = [robot_model.defaultState] * (len(problem.runningModels) + 1)
    us = problem.quasiStatic([robot_model.defaultState] * problem.T)
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

    avrg_dur = sum(duration) / len(duration)
    min_dur = min(duration)
    max_dur = max(duration)
    return avrg_dur, min_dur, max_dur


def runShootingProblemCalcBenchmark(xs, us, problem):
    duration = []
    for _ in range(T):
        c_start = time.time()
        problem.calc(xs, us)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_dur = sum(duration) / len(duration)
    min_dur = min(duration)
    max_dur = max(duration)
    return avrg_dur, min_dur, max_dur


def runShootingProblemCalcDiffBenchmark(xs, us, problem):
    duration = []
    for i in range(T):
        c_start = time.time()
        problem.calcDiff(xs, us)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_dur = sum(duration) / len(duration)
    min_dur = min(duration)
    max_dur = max(duration)
    return avrg_dur, min_dur, max_dur


# Setting up all tasks
if GAIT == "walking":
    GAITPHASE = {
        "walking": {
            "stepLength": 0.25,
            "stepHeight": 0.25,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 2,
        }
    }
elif GAIT == "trotting":
    GAITPHASE = {
        "trotting": {
            "stepLength": 0.15,
            "stepHeight": 0.2,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 2,
        }
    }
elif GAIT == "pacing":
    GAITPHASE = {
        "pacing": {
            "stepLength": 0.15,
            "stepHeight": 0.2,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 2,
        }
    }
elif GAIT == "bounding":
    GAITPHASE = {
        "bounding": {
            "stepLength": 0.007,
            "stepHeight": 0.05,
            "timeStep": 1e-2,
            "stepKnots": 25,
            "supportKnots": 12,
        }
    }
elif GAIT == "jumping":
    GAITPHASE = {
        "jumping": {
            "jumpHeight": 0.15,
            "jumpLength": [0.0, 0.3, 0.0],
            "timeStep": 1e-2,
            "groundKnots": 10,
            "flyingKnots": 20,
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
