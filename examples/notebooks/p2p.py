import example_robot_data
import numpy as np
import pinocchio

import crocoddyl

robot = example_robot_data.load("talos_arm")
robot_model = robot.model

DT = 1e-3
T = 150

state = crocoddyl.StateMultibody(robot.model)

ps = [
    np.array([0.4, 0.0, 0.4]),
    np.array([0.4, 0.4, 0.4]),
    np.array([0.4, 0.4, 0.0]),
    np.array([0.4, 0.0, 0.0]),
]

colors = [
    [1.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, 1.0],
]

cameraTF = [2.0, 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
display = crocoddyl.GepettoDisplay(robot, cameraTF=cameraTF, floor=False)
gv = display.robot.viewer.gui
for i, p in enumerate(ps):
    gv.addSphere(f"world/point{i}", 0.05, colors[i])
    gv.applyConfiguration(f"world/point{i}", [*p.tolist(), 0.0, 0.0, 0.0, 1.0])
gv.refresh()

# State and control regularization costs
xRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelState(state))
uRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state))

# Then let's added the running and terminal cost functions per each action
# model
runningModels = []
terminalModels = []
for p in ps:
    # Create the tracking cost
    goalTrackingCost = crocoddyl.CostModelResidual(
        state,
        crocoddyl.ResidualFrameTranslation(
            state, robot_model.getFrameId("gripper_left_joint"), p
        ),
    )

    actuation = crocoddyl.ActuationModelFull(state)

    # Create the running action model
    runningCostModel = crocoddyl.CostModelSum(state)
    runningCostModel.addCost("gripperPose", goalTrackingCost, 1e-3)
    runningCostModel.addCost("xReg", xRegCost, 1e-4)
    runningCostModel.addCost("uReg", uRegCost, 1e-7)
    runningModels += [
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel
        )
    ]

    # Create the terminal action model
    terminalCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel.addCost("gripperPose", goalTrackingCost, 1.0)
    terminalCostModel.addCost("xreg", xRegCost, 1e-4)
    terminalCostModel.addCost("uReg", uRegCost, 1e-7)
    terminalModels += [
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel
        )
    ]

q0 = np.array([2.0, 1.5, -2.0, 0.0, 0.0, 0.0, 0.0]).T
x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
seqs = [
    [crocoddyl.IntegratedActionModelEuler(runningModel, DT)] * T
    + [crocoddyl.IntegratedActionModelEuler(terminalModel)]
    for runningModel, terminalModel in zip(runningModels, terminalModels)
]
problem = crocoddyl.ShootingProblem(
    x0,
    sum(seqs, [])[:-1],  # noqa: RUF017
    seqs[-1][-1],
)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving it with the DDP algorithm
ddp.solve()

# Penalty if you want.
for i in range(4, 6):
    for m in terminalModels:
        m.costs.costs["gripperPose"].weight = 10**i
    ddp.solve(ddp.xs, ddp.us, 10)

# Visualizing the solution in gepetto-viewer
display.displayFromSolver(ddp)

robot_data = robot_model.createData()
for i, s in enumerate([*seqs, 1]):
    xT = ddp.xs[i * T]
    pinocchio.forwardKinematics(robot_model, robot_data, xT[: state.nq])
    pinocchio.updateFramePlacements(robot_model, robot_data)
    print(
        "Finally reached = ",
        robot_data.oMf[robot_model.getFrameId("gripper_left_joint")].translation.T,
    )
