import numpy as np
import pinocchio as se3
import cddp


class Arm(cddp.NumDiffDynamicalSystem):
  def __init__(self, urdf, path):
    # Getting the Pinocchio model of the robot
    self.robot = se3.robot_wrapper.RobotWrapper(urdf, path)
    self.rmodel = self.robot.model
    self.rdata = self.robot.data

    # Initializing the dynamic model with numerical differentiation
    nq = self.robot.nq + self.robot.nv
    nv = self.robot.nv + self.robot.nv
    m = self.robot.nv
    integrator = cddp.EulerIntegrator()
    discretizer = cddp.EulerDiscretizer()
    cddp.NumDiffDynamicalSystem.__init__(self, nq, nv, m, integrator, discretizer)
  
  def f(self, data, x, u):
    q = x[:self.robot.nq]
    v = x[self.robot.nq:]
    se3.aba(self.rmodel, self.rdata, q, v, u)
    data.f[:self.robot.nq] = v
    data.f[self.robot.nq:] = self.rdata.ddq
    return data.f

  def advanceConfiguration(self, x, dx):
    q = x[:self.robot.nq]
    dq = dx[:self.robot.nv]
    x[:self.robot.nq] = se3.integrate(self.rmodel, q, dq)
    x[self.robot.nq:] += dx[self.robot.nv:]
    return x



# # class SE3Task
import math

class SE3RunningCost(cddp.RunningResidualQuadraticCost):
  def __init__(self, robot, ee_frame, M_des):
    self.robot = robot
    self._frame_idx = self.robot.model.getFrameId(ee_frame)
    self.M_des = M_des
    cddp.RunningResidualQuadraticCost.__init__(self, 6)

  def r(self, data, x, u):
    q = x[:self.robot.nq]
    np.copyto(data.r,
      se3.log(self.M_des.inverse() * self.robot.framePosition(q, self._frame_idx)).vector)
    return data.r

  def rx(self, data, x, u):
    q = x[:self.robot.nq]
    data.rx[:, :self.robot.nq] = \
      se3.jacobian(self.robot.model, self.robot.data, q,
                   self.robot.model.frames[self._frame_idx].parent, False, True)
    return data.rx

  def ru(self, data, x, u):
    return data.ru






np.set_printoptions(linewidth=400, suppress=True, threshold=np.nan)

display = True
if display:
  import gepetto.corbaserver
  cl = gepetto.corbaserver.Client()
  gui = cl.gui


# Creating the system model
import rospkg
path = rospkg.RosPack().get_path('talos_data')
urdf = path + '/robots/talos_left_arm.urdf'
system = Arm(urdf, path)
x0 = np.zeros((system.getConfigurationDimension(), 1))
x0[:system.robot.nq] = np.matrix([ 0.173046, 1., -0.525366, 0., 0., 0.1,-0.005]).T

robot = system.robot
if display:
  if gui.nodeExists("world"):
    gui.deleteNode("world",True)
  robot.initDisplay(loadModel=True)
  robot.display(x0[:robot.nq])

# Defining the SE3 task
frame_name = 'gripper_left_joint'
M_des = se3.SE3(np.eye(3), np.array([ [0.], [0.], [0.4] ]))
se3_cost = SE3RunningCost(system.robot, frame_name, M_des)
w_se3 = np.ones(6)
se3_cost.setWeights(w_se3)

# Defining the velocity and control regularization
from simple_cost import StateControlQuadraticRegularization
xu_reg = StateControlQuadraticRegularization()
wx = 1e-4 * np.hstack([ np.zeros(system.robot.nq), np.ones(system.robot.nv) ])
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
robot = se3.robot_wrapper.RobotWrapper(urdf, path)
frame_idx = robot.model.getFrameId(frame_name)
xf = ddp.intervals[-1].x
qf = xf[:7]
print robot.framePosition(qf, frame_idx)

gui.refresh()
ball_size = 0.04
traj_node = "world/ee_ball"
if gui.nodeExists(traj_node):
  gui.deleteNode(traj_node,True)
gui.addSphere(traj_node, ball_size, [0.,1.,0.,1.])
if display:
  from time import sleep
  robot.initDisplay(loadModel=True)
  it = 0
  for k in ddp.intervals:
    qk = k.x[:robot.nq]
    robot.display(qk)
    M_pos = robot.framePosition(qk, frame_idx)
    gui.applyConfiguration(traj_node, se3.utils.se3ToXYZQUAT(M_pos))
    gui.refresh()

    # gui.addSphere(traj_node+str(it), ball_size/5, [0.5,0.5,0.,1.])
    # gui.applyConfiguration(traj_node+str(it), se3.utils.se3ToXYZQUAT(M_pos))
    sleep(1e-3)
    it += 1