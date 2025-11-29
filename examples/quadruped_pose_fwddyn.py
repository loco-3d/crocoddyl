import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl

if crocoddyl.WITH_ODYN:
    from odyn.utils import plotQPsparsity
from crocoddyl.utils.quadruped import SimpleQuadrupedalGaitProblem, plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)


# Creates a CoM goal problem using terminal constraints
def createCoMGoalProblem(
    model, lfFoot, rfFoot, lhFoot, rhFoot, comGoTo, timeStep, numKnots
):
    gait = SimpleQuadrupedalGaitProblem(model, lfFoot, rfFoot, lhFoot, rhFoot)
    pinocchio.forwardKinematics(gait.rmodel, gait.rdata, q0)
    pinocchio.updateFramePlacements(gait.rmodel, gait.rdata)
    com0 = pinocchio.centerOfMass(gait.rmodel, gait.rdata, q0)
    comForwardModels = [
        gait.createModel(
            timeStep,
            [lfFoot, rfFoot, lhFoot, rhFoot],
        )
        for _ in range(numKnots)
    ]
    amodel = comForwardModels[0]
    nu = amodel.nu
    terminalConstraints = crocoddyl.ConstraintModelManager(gait.state, nu)
    comPosResidual = crocoddyl.ResidualModelCoMPosition(
        gait.state,
        com0 + comGoTo,
        nu,
    )
    eePoseConstraint = crocoddyl.ConstraintModelResidual(gait.state, comPosResidual)
    terminalConstraints.addConstraint("goalCOM", eePoseConstraint)
    dmodel = crocoddyl.DifferentialActionModelContactFwdDynamics(
        gait.state,
        gait.actuation,
        amodel.differential.contacts,
        amodel.differential.costs,
        terminalConstraints,
        0.0,
        True,
    )
    comForwardTermModel = crocoddyl.IntegratedActionModelEuler(dmodel, timeStep)
    # Defining the shooting problem
    return crocoddyl.ShootingProblem(x0, comForwardModels, comForwardTermModel)


# Loading the anymal model
anymal = example_robot_data.load("anymal")

# Defining the initial state of the robot
q0 = anymal.model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(anymal.model.nv)
x0 = np.concatenate([q0, v0])

# Setting up the 3d walking problem
comGoTo = np.array([0.1, 0.0, -0.1])
lfFoot, rfFoot, lhFoot, rhFoot = "LF_FOOT", "RF_FOOT", "LH_FOOT", "RH_FOOT"
problem_1 = createCoMGoalProblem(
    anymal.model, lfFoot, rfFoot, lhFoot, rhFoot, comGoTo, 1e-2, 30
)
problem_2 = createCoMGoalProblem(
    anymal.model, lfFoot, rfFoot, lhFoot, rhFoot, comGoTo, 1e-2, 30
)
solver = crocoddyl.SolverFDDP(problem_1)
if crocoddyl.WITH_ODYN:
    solverSQP = crocoddyl.SolverOdynSQP(problem_2)

# Added the callback functions
if WITHPLOT:
    solver.setCallbacks(
        [
            crocoddyl.CallbackVerbose(),
            crocoddyl.CallbackLogger(),
        ]
    )
    if crocoddyl.WITH_ODYN:
        solverSQP.setCallbacks(
            [crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()]
        )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    if crocoddyl.WITH_ODYN:
        solverSQP.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the OC solver
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
print("*** SOLVE (FeasShoot) ***".format_map(locals()))
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
solver.solve(xs, us, 100, False)
print("*** SOLVE (MultiShoot) ***".format_map(locals()))
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve(xs, us, 100, False)
Ts = int(solver.problem.T / 3)
print("*** SOLVE (HybridShoot: {Ts}) ***".format_map(locals()))
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
solver.solve(xs, us, 100, False)
if crocoddyl.WITH_ODYN:
    print("*** SOLVE (OdynSQP) ***".format_map(locals()))
    solverSQP.solve(xs, us, 100, False)

# Printing the terminal CoM position
np.set_printoptions(precision=4, suppress=True)
print(
    "Target CoM position = ",
    solver.problem.terminalModel.differential.constraints.constraints[
        "goalCOM"
    ].constraint.residual.reference,
)
print(
    "Reached CoM position = ",
    solver.problem.terminalData.differential.multibody.pinocchio.com[0],
)

# Display the entire motion
if WITHDISPLAY:
    display = crocoddyl.MeshcatDisplay(anymal)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        if crocoddyl.WITH_ODYN:
            display.displayFromSolver(solverSQP)
        time.sleep(1.0)

# Plotting the entire motion
if WITHPLOT:
    plotSolution(solver, bounds=False, figIndex=1, show=False)
    log = solver.getCallbacks()[1]
    crocoddyl.plotConvergence(
        log.costs,
        log.pregs,
        log.dregs,
        log.grads,
        log.stops,
        log.steps,
        figIndex=3,
        show=False,
    )
    if crocoddyl.WITH_ODYN:
        plotSolution(solverSQP, figIndex=4, show=False)
        logSQP = solverSQP.getCallbacks()[1]
        crocoddyl.plotConvergence(
            logSQP.costs,
            logSQP.pregs,
            logSQP.dregs,
            logSQP.grads,
            logSQP.stops,
            logSQP.steps,
            figIndex=6,
            show=False,
        )
        plotQPsparsity(solverSQP.qp_model, figIndex=7, show=True)
