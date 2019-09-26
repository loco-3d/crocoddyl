import time

import numpy as np

import crocoddyl
import example_robot_data
import pinocchio
from crocoddyl.utils.biped import SimpleBipedGaitProblem

T = int(5e3)  # number of trials
MAXITER = 1
GAIT = "walking"  # 55 nodes


def runBenchmark(gait_phase):
    robot_model = example_robot_data.loadTalosLegs().model
    rightFoot, leftFoot = 'right_sole_link', 'left_sole_link'
    gait = SimpleBipedGaitProblem(robot_model, rightFoot, leftFoot)
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

    duration = []
    xs = [robot_model.defaultState] * len(ddp.models())
    us = [m.quasiStatic(d, robot_model.defaultState) for m, d in list(zip(ddp.models(), ddp.datas()))[:-1]]
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
            'stepLength': 0.6,
            'stepHeight': 0.1,
            'timeStep': 0.0375,
            'stepKnots': 25,
            'supportKnots': 1
        }
    }

print('cpp-wrapped contact-forward dynamics on quadruped:')
avrg_duration, min_duration, max_duration = runBenchmark(GAITPHASE)
print('  CPU time [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
