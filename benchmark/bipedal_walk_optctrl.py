import crocoddyl
from crocoddyl.utils.biped import SimpleBipedGaitProblem
import pinocchio
import example_robot_data
import numpy as np
import sys
import time

crocoddyl.switchToNumpyMatrix()

T = int(sys.argv[1]) if (len(sys.argv) > 1) else int(5e3)  # number of trials
MAXITER = 1
GAIT = "walking"  # 55 nodes


def createProblem(gait_phase):
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
        problem = gait.createWalkingProblem(x0, value['stepLength'], value['stepHeight'], value['timeStep'],
                                            value['stepKnots'], value['supportKnots'])

    xs = [robot_model.defaultState] * (len(problem.runningModels) + 1)
    us = [
        m.quasiStatic(d, robot_model.defaultState) for m, d in list(zip(problem.runningModels, problem.runningDatas))
    ]
    return xs, us, problem


def runDDPSolveBenchmark(xs, us, problem):
    ddp = crocoddyl.SolverDDP(problem)

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
            'stepLength': 0.6,
            'stepHeight': 0.1,
            'timeStep': 0.0375,
            'stepKnots': 25,
            'supportKnots': 1
        }
    }

print('\033[1m')
print('Python bindings:')
xs, us, problem = createProblem(GAITPHASE)
avrg_duration, min_duration, max_duration = runDDPSolveBenchmark(xs, us, problem)
print('  DDP.solve [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
avrg_duration, min_duration, max_duration = runShootingProblemCalcBenchmark(xs, us, problem)
print('  ShootingProblem.calc [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
avrg_duration, min_duration, max_duration = runShootingProblemCalcDiffBenchmark(xs, us, problem)
print('  ShootingProblem.calcDiff [ms]: {0} ({1}, {2})'.format(avrg_duration, min_duration, max_duration))
print('\033[0m')
