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

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the Borinot quadrotor robot
bluevolta = example_robot_data.load("bluevolta_bravo7_no_ee")
bluevolta.model.gravity.linear[2] = 0.01

# Creating the state and actuaction models
state = crocoddyl.StateMultibody(bluevolta.model)
cf, cm, u_lim, l_lim = 4.138394792004922e-06, 6.991478005829954e-08, 0.0, 20.6991
thrusterNames = [
    "Thruster_BOTTOM_BL",
    "Thruster_BOTTOM_BR",
    "Thruster_BOTTOM_FL",
    "Thruster_BOTTOM_FR",
    "Thruster_TOP_B",
    "Thruster_TOP_FL",
    "Thruster_TOP_FR",
]
ps = []
for name in thrusterNames:
    thrusterId = bluevolta.model.getFrameId(name)
    bMt = bluevolta.model.frames[thrusterId].placement * pinocchio.SE3(
        pinocchio.utils.rpyToMatrix(0.0, np.pi / 2, 0.0), np.zeros(3)
    )
    ps.append(crocoddyl.Thruster(bMt, cm / cf, crocoddyl.ThrusterType.CCW, 0, 150))
actuation = crocoddyl.ActuationModelFloatingBaseThrusters(state, ps)
nv, nu, dt = state.nv, actuation.nu, 3e-2

# Defining the residuals, costs, and constraints
target_pos = np.array([0.5, -1.5, -1.0])
target_quat = pinocchio.Quaternion(pinocchio.utils.rpyToMatrix(0, -np.pi / 2, 0.0))
target_id = state.pinocchio.getFrameId("contact_point")
goalPoseResidual = crocoddyl.ResidualModelFramePlacement(
    state, target_id, pinocchio.SE3(target_quat.matrix(), target_pos), nu
)
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
accResidual = crocoddyl.ResidualModelJointAcceleration(state, nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0.0] * 3 + [0.0] * 3 + [10.0] * (nv - 6) + [10.0] * nv)
)
uResidual = crocoddyl.ResidualModelJointEffort(state, actuation, nu)
xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
uRegCost = crocoddyl.CostModelResidual(state, uResidual)
accRegCost = crocoddyl.CostModelResidual(state, accResidual)
eePoseConstraint = crocoddyl.ConstraintModelResidual(state, goalPoseResidual)

# Adding the costs and constraints
runningCosts = crocoddyl.CostModelSum(state, nu)
terminalCosts = crocoddyl.CostModelSum(state, nu)
terminalConstraints = crocoddyl.ConstraintModelManager(state, nu)
runningCosts.addCost("xReg", xRegCost, 1e-4)
runningCosts.addCost("uReg", uRegCost, 1e-4)
runningCosts.addCost("accReg", accRegCost, 5e-1)
terminalConstraints.addConstraint("goalPose", eePoseConstraint)

# Creating the running and terminal models
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCosts), dt
)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(
        state, actuation, terminalCosts, terminalConstraints
    ),
    dt,
)

# Creating the shooting problem and the OC solver
T = 40
problem_1 = crocoddyl.ShootingProblem(
    np.concatenate([bluevolta.q0, np.zeros(state.nv)]),
    [runningModel] * T,
    terminalModel,
)
problem_2 = crocoddyl.ShootingProblem(
    np.concatenate([bluevolta.q0, np.zeros(state.nv)]),
    [runningModel] * T,
    terminalModel,
)
solver = crocoddyl.SolverFDDP(problem_1)
if crocoddyl.WITH_ODYN:
    solverSQP = crocoddyl.SolverOdynSQP(problem_2)

cameraTF = [-0.03, 4.4, 2.3, -0.02, 0.56, 0.83, -0.03]
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        display = crocoddyl.GepettoDisplay(bluevolta, 4, 4, cameraTF, floor=False)
        bluevolta.viewer.gui.addXYZaxis("world/wp", [1.0, 0.0, 0.0, 1.0], 0.03, 0.5)
        bluevolta.viewer.gui.applyConfiguration(
            "world/wp",
            [
                *target_pos.tolist(),
                target_quat[0],
                target_quat[1],
                target_quat[2],
                target_quat[3],
            ],
        )
        if WITHPLOT:
            solver.setCallbacks(
                [
                    crocoddyl.CallbackVerbose(),
                    crocoddyl.CallbackLogger(),
                    crocoddyl.CallbackDisplay(display),
                ]
            )
            if crocoddyl.WITH_ODYN:
                solverSQP.setCallbacks(
                    [
                        crocoddyl.CallbackVerbose(),
                        crocoddyl.CallbackLogger(),
                        crocoddyl.CallbackDisplay(display),
                    ]
                )
        else:
            solver.setCallbacks(
                [crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)]
            )
            if crocoddyl.WITH_ODYN:
                solverSQP.setCallbacks(
                    [crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)]
                )
    except Exception:
        display = crocoddyl.MeshcatDisplay(bluevolta)
        # display.forceLength = 3
if WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
    if crocoddyl.WITH_ODYN:
        solverSQP.setCallbacks(
            [crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()]
        )
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])
    if crocoddyl.WITH_ODYN:
        solverSQP.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the solver
print("*** SOLVE (FeasShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
solver.solve()
print("*** SOLVE (MultiShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve([], [], 200)
if crocoddyl.WITH_ODYN:
    print("*** SOLVE (OdynSQP) ***")
    solverSQP.solve([], [], 200)

# Printing the terminal pose
np.set_printoptions(precision=4, suppress=True)
Mterm = pinocchio.SE3ToXYZQUAT(
    solver.problem.terminalData.differential.multibody.pinocchio.oMf[target_id]
)
print("Target end-effector pose:")
print("   position:", target_pos)
print("   quaternion:", target_quat.coeffs())
print("Terminal end-effector pose:")
print("   position:", Mterm[:3])
print("   quaternion:", Mterm[3:7])

# Plotting the entire motion
if WITHPLOT:
    log = solver.getCallbacks()[1]
    xs, us = solver.xs, solver.us
    crocoddyl.plotOCSolution(xs, us, figIndex=1, show=False)
    crocoddyl.plotConvergence(
        log.costs,
        log.pregs,
        log.dregs,
        log.stops,
        log.grads,
        log.steps,
        figIndex=2,
        show=False,
    )
    if crocoddyl.WITH_ODYN:
        logSQP = solverSQP.getCallbacks()[1]
        crocoddyl.plotOCSolution(solverSQP.xs, solverSQP.us, figIndex=3, show=False)
        crocoddyl.plotConvergence(
            logSQP.costs,
            logSQP.pregs,
            logSQP.dregs,
            logSQP.stops,
            logSQP.grads,
            logSQP.steps,
            figIndex=4,
            show=False,
        )
        plotQPsparsity(solverSQP.qp_model, figIndex=5, show=True)

# Display the entire motion
if WITHDISPLAY:
    import meshcat.geometry as g

    color = g.MeshLambertMaterial(
        color=display._rgbToHexColor([0.8156, 0.1569, 0.5686, 1.0]),
        reflectivity=0.8,
    )
    display.robot.viewer["target"].set_object(g.Sphere(0.015), color)
    display.robot.viewer["target"].set_transform(
        pinocchio.SE3(target_quat, target_pos).homogeneous
    )
    display.rate = -1
    display.freq = 1
    while True:
        # display.displayFromSolver(solver)
        if crocoddyl.WITH_ODYN:
            display.displayFromSolver(solverSQP)
        time.sleep(1.0)
