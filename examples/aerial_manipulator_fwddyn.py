import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl

WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
signal.signal(signal.SIGINT, signal.SIG_DFL)

# Loading the Borinot quadrotor robot
borinot = example_robot_data.load("borinot")

# Creating the state and actuaction models
state = crocoddyl.StateMultibody(borinot.model)
d1_p, d2_p, d3_p = 0.1602147, 0.0925, 0.185
q1_p, q2_p, q3_p = 0.258819, 0.965926, 0.707107
cf, cm, u_lim, l_lim = 4.138394792004922e-06, 6.991478005829954e-08, 0.0, 20.6991
p1 = pinocchio.SE3(
    pinocchio.Quaternion(np.array([0.0, 0.0, q1_p, q2_p])), np.array([d1_p, d2_p, 0.0])
)
p2 = pinocchio.SE3(
    pinocchio.Quaternion(np.array([0.0, 0.0, q3_p, q3_p])), np.array([0.0, d3_p, 0.0])
)
p3 = pinocchio.SE3(
    pinocchio.Quaternion(np.array([0.0, 0.0, q2_p, q1_p])), np.array([-d1_p, d2_p, 0.0])
)
p4 = pinocchio.SE3(
    pinocchio.Quaternion(np.array([0.0, 0.0, q2_p, -q1_p])),
    np.array([-d1_p, -d2_p, 0.0]),
)
p5 = pinocchio.SE3(
    pinocchio.Quaternion(np.array([0.0, 0.0, -q3_p, q3_p])), np.array([0.0, -d3_p, 0.0])
)
p6 = pinocchio.SE3(
    pinocchio.Quaternion(np.array([0.0, 0.0, -q1_p, q2_p])),
    np.array([d1_p, -d2_p, 0.0]),
)
ps = [
    crocoddyl.Thruster(p1, cm / cf, crocoddyl.ThrusterType.CCW),
    crocoddyl.Thruster(p2, cm / cf, crocoddyl.ThrusterType.CW),
    crocoddyl.Thruster(p3, cm / cf, crocoddyl.ThrusterType.CCW),
    crocoddyl.Thruster(p4, cm / cf, crocoddyl.ThrusterType.CW),
    crocoddyl.Thruster(p5, cm / cf, crocoddyl.ThrusterType.CCW),
    crocoddyl.Thruster(p6, cm / cf, crocoddyl.ThrusterType.CW),
]
actuation = crocoddyl.ActuationModelFloatingBaseThrusters(state, ps)
nv, nu, dt = state.nv, actuation.nu, 3e-2

# Defining the residuals, costs, and constraints
target_pos = np.array([1.0, 0.0, 1.0])
target_quat = pinocchio.Quaternion(
    pinocchio.utils.rpyToMatrix(-np.pi / 2, -np.pi / 2, -np.pi / 6)
)
target_id = state.pinocchio.getFrameId("flying_arm_2__ee")
goalPoseResidual = crocoddyl.ResidualModelFramePlacement(
    state, target_id, pinocchio.SE3(target_quat.matrix(), target_pos), nu
)
xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
accResidual = crocoddyl.ResidualModelJointAcceleration(state, nu)
xActivation = crocoddyl.ActivationModelWeightedQuad(
    np.array([0.1] * 3 + [10.0] * 3 + [10.0] * (nv - 6) + [10.0] * nv)
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
runningCosts.addCost("xReg", xRegCost, 1e-3)
runningCosts.addCost("uReg", uRegCost, 1e-3)
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
T = 33
problem = crocoddyl.ShootingProblem(
    np.concatenate([borinot.q0, np.zeros(state.nv)]), [runningModel] * T, terminalModel
)
solver = crocoddyl.SolverFDDP(problem)

cameraTF = [-0.03, 4.4, 2.3, -0.02, 0.56, 0.83, -0.03]
if WITHDISPLAY:
    try:
        import gepetto

        gepetto.corbaserver.Client()
        display = crocoddyl.GepettoDisplay(borinot, 4, 4, cameraTF, floor=False)
        borinot.viewer.gui.addXYZaxis("world/wp", [1.0, 0.0, 0.0, 1.0], 0.03, 0.5)
        borinot.viewer.gui.applyConfiguration(
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
        else:
            solver.setCallbacks(
                [crocoddyl.CallbackVerbose(), crocoddyl.CallbackDisplay(display)]
            )
    except Exception:
        display = crocoddyl.MeshcatDisplay(borinot)
if WITHPLOT:
    solver.setCallbacks([crocoddyl.CallbackVerbose(), crocoddyl.CallbackLogger()])
else:
    solver.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving the problem with the solver
print("*** SOLVE (FeasShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.FeasShoot)
solver.solve()
print("*** SOLVE (MultiShoot) ***")
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.MultiShoot)
solver.solve()
Ts = int(solver.problem.T / 3)
print("*** SOLVE (HybridShoot: {Ts}) ***".format_map(locals()))
solver.setDynamicsSolver(crocoddyl.DynamicsSolverType.HybridShoot, Ts)
solver.solve()

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
        log.costs, log.u_regs, log.x_regs, log.stops, log.grads, log.steps, figIndex=2
    )

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
        display.displayFromSolver(solver)
        time.sleep(1.0)
