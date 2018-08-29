import cddp
import numpy as np
import pinocchio as se3


np.set_printoptions(linewidth=400, suppress=True, threshold=np.nan)

display = True
plot = True


# Creating the system model
import rospkg
path = rospkg.RosPack().get_path('talos_data')
urdf = path + '/robots/talos_left_arm.urdf'
robot = se3.robot_wrapper.RobotWrapper(urdf, path)
model = robot.model
system = cddp.NumDiffForwardDynamics(model)
x0 = np.zeros((system.getConfigurationDimension(), 1))
x0[:model.nq] = np.matrix([ 0.173046, 1., -0.525366, 0., 0., 0.1,-0.005]).T


# Defining the SE3 task
frame_name = 'gripper_left_joint'
M_des = cddp.se3.SE3(np.eye(3), np.array([ [0.], [0.], [0.5] ]))
se3_cost = cddp.SE3RunningCost(model, frame_name, M_des)
w_se3 = np.ones(6)
se3_cost.setWeights(w_se3)

# Defining the velocity and control regularization
xu_reg = cddp.StateControlQuadraticRegularization()
wx = 1e-4 * np.hstack([ np.zeros(model.nq), np.ones(model.nv) ])
wu = 1e-4 * np.ones(system.getControlDimension())
xu_reg.setWeights(wx, wu)

# Adding the cost functions to the cost manager
cost_manager = cddp.CostManager()
cost_manager.addRunning(xu_reg)
cost_manager.addRunning(se3_cost)

# Setting up the DDP problem
timeline = np.arange(0.0, 0.25, 1e-3)  # np.linspace(0., 0.5, 51)
ddp = cddp.DDP(system, cost_manager, timeline)

# Solving the problem
ddp.compute(x0)

# Printing the final goal
frame_idx = model.getFrameId(frame_name)
xf = ddp.intervals[-1].x
qf = xf[:7]
print robot.framePosition(qf, frame_idx)


if plot:
  X = ddp.getStateTrajectory()
  U = ddp.getControlSequence()
  V = ddp.getTotalCostSequence()
  cddp.plotDDPSolution(model, X, U, V)


if display:
  T = timeline
  X = ddp.getStateTrajectory()
  cddp.visualizePlan(robot, x0, T, X, frame_idx)