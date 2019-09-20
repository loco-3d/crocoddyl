import sys
import time

import example_robot_data
import numpy as np

import crocoddyl
import pinocchio
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem

T = int(5e3)  # number of trials
MAXITER = 1
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


def runBenchmark(gait_phase):
    robot_model = example_robot_data.loadHyQ().model
    lfFoot, rfFoot, lhFoot, rhFoot = 'lf_foot', 'rf_foot', 'lh_foot', 'rh_foot'
    gait = SimpleQuadrupedalGaitProblem(robot_model, lfFoot, rfFoot, lhFoot, rhFoot)
    q0 = robot_model.referenceConfigurations['half_sitting'].copy()
    v0 = pinocchio.utils.zero(robot_model.nv)
    x0 = np.concatenate([q0, v0])

    type_of_gait = list(gait_phase.keys())[0]
    value = gait_phase[type_of_gait]
    if type_of_gait == 'walking':
        # Creating a walking problem
        ddp = crocoddyl.SolverDDP(
            gait.createWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                      value['stepKnots'], value['supportKnots']))
    elif type_of_gait == 'trotting':
        # Creating a trotting problem
        ddp = crocoddyl.SolverDDP(
            gait.createTrottingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                       value['stepKnots'], value['supportKnots']))
    elif type_of_gait == 'pacing':
        # Creating a pacing problem
        ddp = crocoddyl.SolverDDP(
            gait.createPacingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                     value['stepKnots'], value['supportKnots']))
    elif type_of_gait == 'bounding':
        # Creating a bounding problem
        ddp = crocoddyl.SolverDDP(
            gait.createBoundingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                       value['stepKnots'], value['supportKnots']))
    elif type_of_gait == 'jumping':
        # Creating a jumping problem
        ddp = crocoddyl.SolverDDP(gait.createJumpingProblem(x0, value['jumpHeight'], value['timeStep']))

    duration = []
    xs = [robot_model.defaultState] * len(ddp.models())
    us = [m.quasicStatic(d, robot_model.defaultState) for m, d in list(zip(ddp.models(), ddp.datas()))[:-1]]
    for i in range(T):
        c_start = time.time()
        ddp.solve(xs, us, MAXITER, False, 0.1)
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
    GAITPHASE = {'jumping': {'jumpHeight': 0.5, 'timeStep': 1e-2}}

print('cpp-wrapped contact-forward dynamics on quadruped:')
avrg_duration, min_duration, max_duration = runBenchmark(GAITPHASE)
print('  CPU time [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
