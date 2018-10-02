import cddp
import numpy as np
import pinocchio as se3
import os


np.set_printoptions(linewidth=400, suppress=True, threshold=np.nan)

display = True
plot = True


# Reading the URDF file from hyq-description inside our repository
# Note that it redefines the ROS_PACKAGE_PATH in order to be able to find the
# meshes for the Gepetto viewer
filename = str(os.path.dirname(os.path.abspath(__file__)))
os.environ['ROS_PACKAGE_PATH'] = filename
path = filename + '/hyq_description/'
urdf = path + 'robots/hyq_no_sensors.urdf'

# Getting the robot model from the URDF file
robot = se3.robot_wrapper.RobotWrapper(urdf, path, se3.JointModelFreeFlyer())
model = robot.model

# Creating the system model
system = cddp.NumDiffSparseConstrainedForwardDynamics(model)

# Initial state
q0 = robot.q0
q0[7] = -0.2*0.
q0[7+1] = 0.75
q0[7+2] = -1.5
q0[7+3] = -0.2*0.
q0[7+4] = -0.75
q0[7+5] = 1.5
q0[7+6] = -0.2*0.
q0[7+7] = 0.75
q0[7+8] = -1.5
q0[7+9] = -0.2*0.
q0[7+10] = -0.75
q0[7+11] = 1.5
v0 = np.zeros((system.getTangentDimension(), 1))
u0 = np.zeros((system.getTangentDimension(), 1))
x0 = np.vstack([q0, v0])


# Defining the SE3 task
frame_name = 'base_link'
M_des = cddp.se3.SE3(np.eye(3), np.array([ [0.1], [0.], [0.] ]))
se3_rcost = cddp.SE3RunningCost(model, frame_name, M_des)
se3_tcost = cddp.SE3TerminalCost(model, frame_name, M_des)
w_se3 = np.array([1., 1., 1., 1., 1., 1.])
se3_rcost.setWeights(1000*w_se3)
se3_tcost.setWeights(1000*w_se3)

# Defining the CoM task
com_des = np.matrix([ [0.1], [0.], [0.] ])
com_cost = cddp.CoMRunningCost(model, com_des)
w_com = 1000.*np.array([1., 1., 1.])
com_cost.setWeights(w_com)


# Defining the velocity and control regularization
xu_reg = cddp.StateControlQuadraticRegularization()
wx = 1e-4 * np.hstack([ np.zeros(model.nv), np.ones(model.nv) ])
wu = 1e-4 * np.ones(system.getControlDimension())
xu_reg.setWeights(wx, wu)

# Adding the cost functions to the cost manager
cost_manager = cddp.CostManager()
cost_manager.addRunning(xu_reg)
cost_manager.addRunning(se3_rcost)
cost_manager.addTerminal(se3_tcost)
# cost_manager.addRunning(com_cost)

# Setting up the DDP problem
timeline = np.arange(0.0, 0.25, 1e-3)  # np.linspace(0., 0.5, 51)
ddp = cddp.DDP(system, cost_manager, timeline, cddp.DDPDebug(robot))

# Configuration the solver from YAML file
ddp.setFromConfigFile(filename + "/hyq_config.yaml")

# Solving the problem
ddp.compute(x0)


# Printing the final goal
frame_idx = model.getFrameId(frame_name)
xf = ddp.intervals[-1].x
qf = xf[:model.nq]
print robot.framePosition(qf, frame_idx)


if plot:
  J = ddp.getTotalCostSequence()
  gamma, theta, alpha = ddp.getConvergenceSequence()
  cddp.plotDDPConvergence(J, gamma, 1e-3*theta, alpha)

if display:
  T = timeline
  X = ddp.getStateTrajectory()
  cddp.visualizePlan(robot, T, x0, X)

# ddp.saveToFile('mu_1e2.txt')