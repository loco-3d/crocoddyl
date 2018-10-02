import cddp
import numpy as np
import pinocchio as se3
import os


np.set_printoptions(linewidth=400, suppress=True, threshold=np.nan)

display = True
plot = True
constraint = False


# Creating the system model
import rospkg
filename = str(os.path.dirname(os.path.abspath(__file__)))
path = rospkg.RosPack().get_path('talos_data')
urdf = path + '/robots/talos_left_arm.urdf'
robot = se3.robot_wrapper.RobotWrapper(urdf, path)
model = robot.model
# system = cddp.NumDiffForwardDynamics(model)
# system = cddp.NumDiffSparseForwardDynamics(model)
system = cddp.SparseForwardDynamics(model)

# Initial state
q0 = np.matrix([ 0.173046, 1., -0.525366, 0., 0., 0.1,-0.005]).T
v0 = np.zeros((system.getTangentDimension(), 1))
x0 = np.vstack([q0, v0])


# Defining the SE3 task
frame_name = 'gripper_left_joint'
t_des = np.array([ [0.], [0.], [0.4] ])
R_des = np.eye(3)
M_des = cddp.se3.SE3(R_des, t_des)
se3_rcost = cddp.SE3RunningCost(model, frame_name, M_des)
se3_tcost = cddp.SE3TerminalCost(model, frame_name, M_des)
w_se3 = np.array([1., 1., 1., 1., 1., 1.])
se3_rcost.setWeights(w_se3)
se3_tcost.setWeights(w_se3)

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

# Setting up the DDP problem
timeline = np.arange(0.0, 0.25, 1e-3)  # np.linspace(0., 0.5, 51)
ddp = cddp.DDP(system, cost_manager, timeline, cddp.DDPDebug(robot))

# Configuration the solver from YAML file
ddp.setFromConfigFile(filename + "/arm_config.yaml")

# Solving the problem
ddp.compute(x0)


# Printing the final goal
frame_idx = model.getFrameId(frame_name)
xf = ddp.intervals[-1].x
qf = xf[:7]
print robot.framePosition(qf, frame_idx)

# ddp.computeGradientNumerically()


if plot:
  J = ddp.getTotalCostSequence()
  gamma, theta, alpha = ddp.getConvergenceSequence()
  cddp.plotDDPConvergence(J, gamma, theta, alpha)

if display:
  T = timeline
  X = ddp.getStateTrajectory()
  cddp.visualizePlan(robot, T, x0, X, frame_idx)


if constraint:
  # Defining the joint limits as soft-constraint
  jpos_lim = cddp.JointPositionBarrier(model)
  cost_manager.addRunning(jpos_lim)
  
  # Solving the problem with constraints
  ddp = cddp.DDP(system, cost_manager, timeline)
  ddp.compute(x0, ddp.getControlSequence())

  if plot:
    J = ddp.getTotalCostSequence()
    gamma, theta, alpha = ddp.getConvergenceSequence()
    cddp.plotDDPConvergence(J, gamma, theta, alpha)

  if display:
    X = ddp.getStateTrajectory()
    cddp.visualizePlan(robot, T, x0, X, frame_idx)