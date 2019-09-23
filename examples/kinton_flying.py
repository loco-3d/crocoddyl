ºfrom crocoddyl import *
import pinocchio as pin
import numpy as np
from crocoddyl.diagnostic import displayTrajectory

# LOAD ROBOT
robot = loadKinton()
robot.initDisplay(loadModel=True)
robot.display(robot.q0)

robot.framesForwardKinematics(robot.q0)

# DEFINE TARGET POSITION
target_pos  = np.array([0,0,3])
target_quat = pin.Quaternion(0, 0, 0, 1)
target_quat.normalize()

# Plot goal frame
robot.viewer.gui.addXYZaxis('world/framegoal', [1., 0., 0., 1.], .015, 4)
robot.viewer.gui.applyConfiguration('world/framegoal', target_pos.tolist() + [target_quat[0], target_quat[1], target_quat[2], target_quat[3]])
robot.viewer.gui.refresh()

# ACTUATION MODEL
actModel = ActuationModelUAM(robot.model)

# COST MODEL
# Create a cost model per the running and terminal action model.
runningCostModel = CostModelSum(robot.model, actModel.nu)
terminalCostModel = CostModelSum(robot.model, actModel.nu)

frameName = 'base_link'
state = StatePinocchio(robot.model)
SE3ref = pin.SE3()
SE3ref.translation = target_pos.reshape(3,1)
SE3ref.rotation = target_quat.matrix()


stateWeights = np.array([0] * 3 + [500.] * 3 + [0.01] * (robot.model.nv - 2) + [10] * robot.model.nv)
goalTrackingCost = CostModelFramePlacement(robot.model,
                                           frame=robot.model.getFrameId(frameName),
                                           ref=SE3ref,
                                           nu =actModel.nu)
xRegCost = CostModelState(robot.model, state, ref=state.zero(), nu=actModel.nu)
uRegCost = CostModelControl(robot.model, nu=robot.model.nv-2)

# Then let's add the running and terminal cost functions
runningCostModel.addCost(name="pos", weight=1e-3, cost=goalTrackingCost)
runningCostModel.addCost(name="regx", weight=1e-7, cost=xRegCost)
runningCostModel.addCost(name="regu", weight=1e-7, cost=uRegCost)
terminalCostModel.addCost(name="pos", weight=50, cost=goalTrackingCost)

# DIFFERENTIAL ACTION MODEL
runningModel = IntegratedActionModelEuler(DifferentialActionModelUAM(robot.model, actModel, runningCostModel))
terminalModel = IntegratedActionModelEuler(DifferentialActionModelUAM(robot.model, actModel, terminalCostModel))

# DEFINING THE SHOOTING PROBLEM & SOLVING

# Defining the time duration for running action models and the terminal one
dt = 1e-3
runningModel.timeStep = dt

# For this optimal control problem, we define 250 knots (or running action
# models) plus a terminal knot
T = 250
q0 = [0., 0., 0., 0., 0., 0.]
q0 = robot.model.referenceConfigurations["initial_pose"]

x0 = np.hstack([m2a(q0), np.zeros(robot.model.nv)])
problem = ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPVerbose()]
ddp.callback.append(CallbackDDPLogger())

# Solving it with the DDP algorithm
ddp.solve()

displayTrajectory(robot, ddp.xs, runningModel.timeStep)
