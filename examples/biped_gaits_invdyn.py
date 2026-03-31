import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.biped import SimpleBipedGaitProblem, plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Creating the lower-body part of Talos
talos_legs = example_robot_data.load("talos_legs")
talos_legs.model.velocityLimit *= np.inf

# Defining the initial state of the robot
q0 = talos_legs.model.referenceConfigurations["half_sitting"].copy()
v0 = pinocchio.utils.zero(talos_legs.model.nv)
x0 = np.concatenate([q0, v0])
x0_SQP = np.concatenate([q0, v0])

# Setting up the 3d walking problem
rightFoot = "right_sole_link"
leftFoot = "left_sole_link"
gait = SimpleBipedGaitProblem(talos_legs.model, rightFoot, leftFoot, fwddyn=False)

# Setting up all tasks
GAITPHASES = [
    {
        "walking": {
            "stepLength": 0.6,
            "stepHeight": 0.1,
            "timeStep": 0.03,
            "stepKnots": 15,
            "supportKnots": 4,
        }
    },
    {
        "jumping": {
            "jumpHeight": 0.1,
            "jumpLength": [0.0, 0.3, 0.0],
            "timeStep": 0.03,
            "groundKnots": 9,
            "flyingKnots": 6,
        }
    },
    {
        "walking": {
            "stepLength": 0.6,
            "stepHeight": 0.1,
            "timeStep": 0.03,
            "stepKnots": 15,
            "supportKnots": 2,
        }
    },
]

solver = [None] * len(GAITPHASES)
solverSQP = [None] * len(GAITPHASES)
for i, phase in enumerate(GAITPHASES):
    for key, value in phase.items():
        if key == "walking":
            # Creating a walking problem
            solver[i] = crocoddyl.SolverIntro(
                gait.createWalkingProblem(
                    x0,
                    value["stepLength"],
                    value["stepHeight"],
                    value["timeStep"],
                    value["stepKnots"],
                    value["supportKnots"],
                )
            )
            if crocoddyl.WITH_ODYN:
                solverSQP[i] = crocoddyl.SolverOdynSQP(
                    gait.createWalkingProblem(
                        x0_SQP,
                        value["stepLength"],
                        value["stepHeight"],
                        value["timeStep"],
                        value["stepKnots"],
                        value["supportKnots"],
                        constraint=True,
                    )
                )
        elif key == "jumping":
            # Creating a jumping problem
            solver[i] = crocoddyl.SolverIntro(
                gait.createJumpingProblem(
                    x0,
                    value["jumpHeight"],
                    value["jumpLength"],
                    value["timeStep"],
                    value["groundKnots"],
                    value["flyingKnots"],
                )
            )
            if crocoddyl.WITH_ODYN:
                solverSQP[i] = crocoddyl.SolverOdynSQP(
                    gait.createJumpingProblem(
                        x0_SQP,
                        value["jumpHeight"],
                        value["jumpLength"],
                        value["timeStep"],
                        value["groundKnots"],
                        value["flyingKnots"],
                        constraint=True,
                    )
                )

    # Added the callback functions
    if WITHPLOT:
        solver[i].setCallbacks(
            [
                crocoddyl.CallbackVerbose(),
                crocoddyl.CallbackLogger(),
            ]
        )
        if crocoddyl.WITH_ODYN:
            solverSQP[i].setCallbacks(
                [crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()]
            )
    else:
        solver[i].setCallbacks([crocoddyl.CallbackVerbose()])
        if crocoddyl.WITH_ODYN:
            solverSQP[i].setCallbacks([crocoddyl.CallbackVerbose()])

    # Solving the problem with the OC solver
    xs = [x0] * (solver[i].problem.T + 1)
    us = solver[i].problem.quasiStatic([x0] * solver[i].problem.T)
    print("*** SOLVE {key} (FeasShoot) ***".format_map(locals()))
    solver[i].setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
    solver[i].solve(xs, us, 100, False)
    print("*** SOLVE {key} (MultiShoot) ***".format_map(locals()))
    solver[i].setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
    solver[i].solve(xs, us, 100, False)
    Ts = int(solver[i].problem.T / 3)
    print("*** SOLVE {key} (HybridShoot: {Ts}) ***".format_map(locals()))
    solver[i].setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
    solver[i].solve(xs, us, 100, False)
    if crocoddyl.WITH_ODYN:
        xs_sqp = [x0_SQP] * (solverSQP[i].problem.T + 1)
        us_sqp = solverSQP[i].problem.quasiStatic([x0_SQP] * solverSQP[i].problem.T)
        print("*** SOLVE {key} (OdynSQP) ***".format_map(locals()))
        solverSQP[i].solve(xs, us, 100, False)
    if i != len(GAITPHASES) - 1:
        print()

    # Defining the final state as initial one for the next phase
    x0 = solver[i].xs[-1]
    if crocoddyl.WITH_ODYN:
        x0_SQP = solverSQP[i].xs[-1]

# Display the entire motion
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        cameraTF = [3.0, 3.68, 0.84, 0.2, 0.62, 0.72, 0.22]
        display = crocoddyl.GepettoDisplay(talos_legs, 4, 4, cameraTF)
    except Exception:
        display = crocoddyl.MeshcatDisplay(talos_legs)
    display.rate = -1
    display.freq = 1
    while True:
        for i, phase in enumerate(GAITPHASES):
            display.displayFromSolver(solver[i])
        if crocoddyl.WITH_ODYN:
            for i, phase in enumerate(GAITPHASES):
                display.displayFromSolver(solverSQP[i])
        time.sleep(1.0)

# Plotting the entire motion
if WITHPLOT:
    plotSolution(solver, bounds=False, figIndex=1, show=False)
    for i, phase in enumerate(GAITPHASES):
        title = next(iter(phase.keys())) + " (phase " + str(i) + ")"
        log = solver[i].getCallbacks()[1]
        fig_idx = i + 3
        crocoddyl.plotConvergence(
            log.costs,
            log.pregs,
            log.dregs,
            log.grads,
            log.stops,
            log.steps,
            figTitle=title,
            figIndex=i + 3,
            show=False,
        )
    if crocoddyl.WITH_ODYN:
        plotSolution(solverSQP, figIndex=fig_idx + 1, show=False)
        for i, phase in enumerate(GAITPHASES):
            title = next(iter(phase.keys())) + " (phase " + str(i) + ")"
            log = solverSQP[i].getCallbacks()[1]
            crocoddyl.plotConvergence(
                log.costs,
                log.pregs,
                log.dregs,
                log.grads,
                log.stops,
                log.steps,
                figTitle=title,
                figIndex=i + fig_idx + 3,
                show=True if i == len(GAITPHASES) - 1 else False,
            )
