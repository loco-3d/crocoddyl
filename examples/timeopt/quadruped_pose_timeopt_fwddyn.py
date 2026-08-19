import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl
from crocoddyl.utils.quadruped import plotSolution

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the B1 quadruped model
robot = example_robot_data.load("b1")

# Defining the initial state of the robot
q0 = robot.model.referenceConfigurations["standing"].copy()
v0 = pinocchio.utils.zero(robot.model.nv)
x0 = np.concatenate([q0, v0])
robot.model.defaultState = x0

# Creating the state and actuation models
state = crocoddyl.StateMultibody(robot.model)
actuation = crocoddyl.ActuationModelMultibody(state)
nu, dt = actuation.nu, 1e-2
np_total = 1
time_reg_weight = 3e2
runningTime = crocoddyl.IntegratorTime(dt, True)
terminalTime = runningTime

# Defining contact and CoM references
foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
pinocchio.forwardKinematics(robot.model, robot.data, q0)
pinocchio.updateFramePlacements(robot.model, robot.data)
com_ref = pinocchio.centerOfMass(robot.model, robot.data, q0) + np.array(
    [0.1, 0.0, 0.0]
)

# Defining the residuals, costs, and constraints
contactConstraints = crocoddyl.ImplicitConstraintModelMultiple(state, nu)
for name in foot_names:
    frame_id = robot.model.getFrameId(name)
    contact = crocoddyl.ContactModel(
        state,
        frame_id,
        pinocchio.SE3.Identity(),
        pinocchio.LOCAL_WORLD_ALIGNED,
        nu,
        np.array([0.0, 50.0]),
        [True, True, True, False, False, False],
    )
    contactConstraints.addConstraint(name + "_contact", contact)

runningCosts = crocoddyl.CostModelSum(state, nu, np_total)
terminalCosts = crocoddyl.CostModelSum(state, nu, np_total)
terminalConstraints = crocoddyl.ConstraintModelManager(state, nu)

mu = 0.7
Rsurf = np.eye(3)
for name in foot_names:
    frame_id = robot.model.getFrameId(name)
    cone = crocoddyl.FrictionCone(Rsurf, mu, 4, False)
    coneResidual = crocoddyl.ResidualModelContactFrictionCone(
        state, frame_id, cone, nu, True
    )
    coneActivation = crocoddyl.ActivationModelQuadraticBarrier(
        crocoddyl.ActivationBounds(cone.lb, cone.ub)
    )
    frictionCone = crocoddyl.CostModelResidual(state, coneActivation, coneResidual)
    runningCosts.addCost(name + "_frictionCone", frictionCone, 1e1)

stateWeights = np.array(
    [0.0] * 3
    + [500.0] * 3
    + [0.01] * (robot.model.nv - 6)
    + [10.0] * 6
    + [1.0] * (robot.model.nv - 6)
)
stateResidual = crocoddyl.ResidualModelState(state, robot.model.defaultState, nu)
stateActivation = crocoddyl.ActivationModelWeightedQuad(stateWeights**2)
ctrlResidual = crocoddyl.ResidualModelControl(state, nu)
pResidual = crocoddyl.ResidualModelParameters(state, np.array([np.log(dt)]), nu)
comResidual = crocoddyl.ResidualModelCoMPosition(state, com_ref, nu)
stateReg = crocoddyl.CostModelResidual(state, stateActivation, stateResidual)
ctrlReg = crocoddyl.CostModelResidual(state, ctrlResidual)
pReg = crocoddyl.CostModelResidual(state, pResidual)
comConstraint = crocoddyl.ConstraintModelResidual(state, comResidual)

runningCosts.addCost("stateReg", stateReg, 1e1)
runningCosts.addCost("ctrlReg", ctrlReg, 1e-1)
runningCosts.addCost("pReg", pReg, time_reg_weight)
terminalConstraints.addConstraint("goalCOM", comConstraint)

# Creating the running and terminal models
runningDynamics = crocoddyl.DynamicsModelConstrainedForward(
    state, actuation, contactConstraints, np_total
)
terminalDynamics = crocoddyl.DynamicsModelConstrainedForward(
    state, actuation, contactConstraints, np_total
)
runningModel = crocoddyl.IntegratedActionModelEuler(
    runningDynamics,
    runningCosts,
    None,
    None,
    runningTime,
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    terminalDynamics,
    terminalCosts,
    terminalConstraints,
    None,
    terminalTime,
)

# Creating the parameter manager, shooting problem, and solver
T = 100
p0 = np.array([np.log(dt)])
params = crocoddyl.ParameterManager(state)
params.addParam("timeopt", crocoddyl.IntegratorTimeoptParams(state, runningTime))
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel, params)
problem.update_p(p0, phase_idx=0)
solver = crocoddyl.SolverFDDP(problem)
solver.th_minImprove = 1e-4
solver.th_stop = 1e-4
if WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the FDDP solver
xs = [x0] * (solver.problem.T + 1)
us = solver.problem.quasiStatic([x0] * solver.problem.T)
print("*** SOLVE (FeasShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
solver.solve(xs, us, [p0], 200, False)
print("*** SOLVE (MultiShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve(xs, us, [p0], 200, False)
Ts = int(solver.problem.T / 3)
print("*** SOLVE (HybridShoot: {Ts}) ***".format_map(locals()))
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
solver.solve(xs, us, [p0], 200, False)

print("step integration (guess): ", dt)
print("step integration (optimal): ", np.exp(solver.p[0][0]))

np.set_printoptions(precision=4, suppress=True)
print("Target CoM position = ", com_ref)
pinocchio_data = robot.model.createData()
reached_com = pinocchio.centerOfMass(
    robot.model, pinocchio_data, solver.xs[-1][: robot.model.nq]
)
print("Reached CoM position = ", reached_com)

if WITHDISPLAY:
    display = crocoddyl.MeshcatDisplay(robot)
    display.rate = -1
    display.freq = 1
    while True:
        display.displayFromSolver(solver)
        time.sleep(1.0)

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
