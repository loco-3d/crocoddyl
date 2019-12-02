import crocoddyl
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem
import pinocchio
import example_robot_data
import numpy as np
import os
import sys
import time
import subprocess

crocoddyl.switchToNumpyMatrix()

T = int(sys.argv[1]) if (len(sys.argv) > 1) else int(5e3)  # number of trials
MAXITER = 1
CALLBACKS = False
WALKING = 'walk' in sys.argv
TROTTING = 'trot' in sys.argv
PACING = 'pace' in sys.argv
BOUNDING = 'bound' in sys.argv
JUMPING = 'jump' in sys.argv

GAIT = "walking"  # 104 nodes
if WALKING:
    print('running walking benchmark ...')
    GAIT = "walking"  # 104 nodes
if TROTTING:
    print('running trotting benchmark ...')
    GAIT = "trotting"  # 54 nodes
if PACING:
    print('running pacing benchmark ...')
    GAIT = "pacing"  # 54 nodes
if BOUNDING:
    print('running bounding benchmark ...')
    GAIT = "bounding"  # 54 nodes
if JUMPING:
    print('running jumping benchmark ...')
    GAIT = "jumping"  # 61 nodes


def createProblem(gait_phase):
    robot_model = example_robot_data.loadHyQ().model
    lfFoot, rfFoot, lhFoot, rhFoot = 'lf_foot', 'rf_foot', 'lh_foot', 'rh_foot'
    gait = SimpleQuadrupedalGaitProblem(robot_model, lfFoot, rfFoot, lhFoot, rhFoot)
    q0 = robot_model.referenceConfigurations['standing'].copy()
    v0 = pinocchio.utils.zero(robot_model.nv)
    x0 = np.concatenate([q0, v0])

    type_of_gait = list(gait_phase.keys())[0]
    value = gait_phase[type_of_gait]
    if type_of_gait == 'walking':
        # Creating a walking problem
        problem = gait.createWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                            value['stepKnots'], value['supportKnots'])
    elif type_of_gait == 'trotting':
        # Creating a trotting problem
        problem = gait.createTrottingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                             value['stepKnots'], value['supportKnots'])
    elif type_of_gait == 'pacing':
        # Creating a pacing problem
        problem = gait.createPacingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                           value['stepKnots'], value['supportKnots'])
    elif type_of_gait == 'bounding':
        # Creating a bounding problem
        problem = gait.createBoundingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                             value['stepKnots'], value['supportKnots'])
    elif type_of_gait == 'jumping':
        # Creating a jumping problem
        problem = gait.createJumpingProblem(x0, value['jumpHeight'], value['jumpLength'], value['timeStep'],
                                            value['groundKnots'], value['flyingKnots'])

    xs = [robot_model.defaultState] * (len(problem.runningModels) + 1)
    us = [
        m.quasiStatic(d, robot_model.defaultState) for m, d in list(zip(problem.runningModels, problem.runningDatas))
    ]
    return xs, us, problem


def runDDPSolveBenchmark(xs, us, problem):
    ddp = crocoddyl.SolverFDDP(problem)
    if CALLBACKS:
        ddp.setCallbacks([crocoddyl.CallbackVerbose()])
    duration = []
    for i in range(T):
        c_start = time.time()
        ddp.solve(xs, us, MAXITER, False, 0.1)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_duration = sum(duration) / len(duration)
    min_duration = min(duration)
    max_duration = max(duration)
    return avrg_duration, min_duration, max_duration


def runShootingProblemCalcBenchmark(xs, us, problem):
    duration = []
    for i in range(T):
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
    for i in range(T):
        c_start = time.time()
        problem.calcDiff(xs, us)
        c_end = time.time()
        duration.append(1e3 * (c_end - c_start))

    avrg_duration = sum(duration) / len(duration)
    min_duration = min(duration)
    max_duration = max(duration)
    return avrg_duration, min_duration, max_duration


# Setting up all tasks
if GAIT == 'walking':
    GAITPHASE = {
        'walking': {
            'stepLength': 0.25,
            'stepHeight': 0.25,
            'timeStep': 1e-2,
            'stepKnots': 25,
            'supportKnots': 2
        }
    }
elif GAIT == 'trotting':
    GAITPHASE = {
        'trotting': {
            'stepLength': 0.15,
            'stepHeight': 0.2,
            'timeStep': 1e-2,
            'stepKnots': 25,
            'supportKnots': 2
        }
    }
elif GAIT == 'pacing':
    GAITPHASE = {
        'pacing': {
            'stepLength': 0.15,
            'stepHeight': 0.2,
            'timeStep': 1e-2,
            'stepKnots': 25,
            'supportKnots': 2
        }
    }
elif GAIT == 'bounding':
    GAITPHASE = {
        'bounding': {
            'stepLength': 0.007,
            'stepHeight': 0.05,
            'timeStep': 1e-2,
            'stepKnots': 25,
            'supportKnots': 12
        }
    }
elif GAIT == 'jumping':
    GAITPHASE = {
        'jumping': {
            'jumpHeight': 0.15,
            'jumpLength': [0.0, 0.3, 0.],
            'timeStep': 1e-2,
            'groundKnots': 10,
            'flyingKnots': 20
        }
    }

print('\033[1m')
print('C++:')
popen = subprocess.check_call([os.path.dirname(os.path.abspath(__file__)) + "/quadrupedal-gaits", str(T)])

print('Python bindings:')
xs, us, problem = createProblem(GAITPHASE)
avrg_duration, min_duration, max_duration = runDDPSolveBenchmark(xs, us, problem)
print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
avrg_duration, min_duration, max_duration = runShootingProblemCalcBenchmark(xs, us, problem)
print('  ShootingProblem.calc [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
avrg_duration, min_duration, max_duration = runShootingProblemCalcDiffBenchmark(xs, us, problem)
print('  ShootingProblem.calcDiff [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
print('\033[0m')
