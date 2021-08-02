import crocoddyl
import pinocchio
import numpy as np
import example_robot_data

robot = example_robot_data.load('talos_arm')
robot_model = robot.model

DT = 1e-3
T = 250
target = np.array([0.4, 0., .4])

cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
display = crocoddyl.GepettoDisplay(robot, cameraTF=cameraTF, floor=False)
robot.viewer.gui.addSphere('world/point', .05, [1., 0., 0., 1.])  # radius = .1, RGBA=1001
robot.viewer.gui.applyConfiguration('world/point', target.tolist() + [0., 0., 0., 1.])  # xyz+quaternion
robot.viewer.gui.refresh()

# Create the cost functions
state = crocoddyl.StateMultibody(robot.model)
goalTrackingCost = crocoddyl.CostModelFrameTranslation(state, robot_model.getFrameId("gripper_left_joint"), target)
xRegCost = crocoddyl.CostModelState(state)
uRegCost = crocoddyl.CostModelControl(state)

# Create cost model per each action model
runningCostModel = crocoddyl.CostModelSum(state)
terminalCostModel = crocoddyl.CostModelSum(state)

# Then let's added the running and terminal cost functions
runningCostModel.addCost("gripperPose", goalTrackingCost, 1.)
runningCostModel.addCost("stateReg", xRegCost, 5e-2)
runningCostModel.addCost("ctrlReg", uRegCost, 1e-5)
terminalCostModel.addCost("gripperPose", goalTrackingCost, 10000.)
terminalCostModel.addCost("stateReg", xRegCost, 5e-2)
terminalCostModel.addCost("ctrlReg", uRegCost, 1e-5)

# Create the action model
actuation = crocoddyl.ActuationModelFull(state)
runningModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel), DT)
terminalModel = crocoddyl.IntegratedActionModelEuler(
    crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel))
#runningModel.differential.armature = 0.2 * np.matrix(np.ones(state.nv)).T
#terminalModel.differential.armature = 0.2 * np.matrix(np.ones(state.nv)).T

# Create the problem
q0 = np.array([2., 1.5, -2., 0., 0., 0., 0.]).T
x0 = np.concatenate([q0, pinocchio.utils.zero(state.nv)])
problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackVerbose()])

# Solving it with the DDP algorithm
ddp.solve()

# Visualizing the solution in gepetto-viewer
display.displayFromSolver(ddp)

robot_data = robot_model.createData()
xT = ddp.xs[-1]
pinocchio.forwardKinematics(robot_model, robot_data, xT[:state.nq])
pinocchio.updateFramePlacements(robot_model, robot_data)
print('Finally reached = ', robot_data.oMf[robot_model.getFrameId("gripper_left_joint")].translation.T)
