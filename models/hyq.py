import cddp
import numpy as np
import pinocchio as se3
import os

np.set_printoptions(linewidth=400, suppress=False, threshold=np.nan)

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

timeline = np.arange(0.0, 0.25, 1e-3)  # np.linspace(0., 0.5, 51)

# Create the contact information
contact_indices = [robot.model.getFrameId("lf_foot"),
                   robot.model.getFrameId("rf_foot"),
                   robot.model.getFrameId("rh_foot")]
contactPhase0 = cddp.multiphase.Phase(contact_indices, 0.,np.inf)
contactInfo = cddp.multiphase.Multiphase([contactPhase0], 3)

# Create the ddp dynamics
ddpDynamics = cddp.dynamics.FloatingBaseMultibodyDynamics(robot.model, contactInfo)

# Create the integration and dynamics derivatives schemes.
ddpIntegrator = cddp.system.integrator.FloatingBaseMultibodyEulerIntegrator()
ddpDiscretizer = cddp.system.discretizer.FloatingBaseMultibodyEulerExpDiscretizer()

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
v0 = np.zeros((robot.model.nv, 1))
U0 = [np.zeros((robot.model.nv-6, 1)) for i in xrange(len(timeline)-1)]
x0 = np.vstack([q0, v0])


# Defining the SE3 task
frame_name = 'base_link'
M_des = se3.SE3(np.eye(3), np.array([ [0.1], [0.], [0.] ]))
"""
se3_rcost = cddp.tasks.multibody_tasks.SE3RunningCost(model, frame_name, M_des)
se3_tcost = cddp.SE3TerminalCost(model, frame_name, M_des)
w_se3 = np.array([1., 1., 1., 1., 1., 1.])
se3_rcost.setWeights(1000*w_se3)
se3_tcost.setWeights(1000*w_se3)
"""
w_se3 = np.ones((6,1))
se3_cost = cddp.costs.multibody_dynamics.SE3Cost(ddpDynamics, M_des, 1000.*w_se3, frame_name)

# Defining the CoM task
com_des = np.matrix([ [0.1], [0.], [0.] ])
w_com = 1000.*np.ones((3,1))
com_cost = cddp.costs.multibody_dynamics.CoMCost(ddpDynamics, com_des, w_com)


# Defining the velocity and control regularization
wx = 1e-4 * np.vstack([ np.zeros((model.nv,1)), np.ones((model.nv,1)) ])
wu = 1e-4 * np.ones((robot.nv-6,1))

#TODO: Why are we regularizing to zero posture!
x_cost = cddp.costs.multibody_dynamics.StateCost(ddpDynamics,
                                                 x0, wx)
u_cost = cddp.costs.multibody_dynamics.ControlCost(ddpDynamics,
                                                   np.zeros((robot.model.nv-6,1)), wu)

# Adding the cost functions to the cost manager
costManager = cddp.cost_manager.CostManager(ddpDynamics)
costManager.addRunning(x_cost)
costManager.addRunning(u_cost)
costManager.addRunning(se3_cost)
costManager.addTerminal(se3_cost)
#costManager.addRunning(com_cost)

# Setting up the DDP problem

ddpModel = cddp.ddp_model.DDPModel(ddpDynamics, ddpIntegrator,
                                   ddpDiscretizer, costManager)
ddpData = cddp.ddp_model.DDPData(ddpModel, timeline)

#TODO: Move to proper location
ddpModel.eps = 1e-8

# Configuration the solver from YAML file
solverParams = cddp.solver.SolverParams()
solverParams.setFromConfigFile(filename + "/hyq_config.yaml")

# Solving the problem
#ddp.compute(x0)
cddp.Solver.setInitial(ddpModel, ddpData, xInit=x0, UInit=U0)
cddp.Solver.solve(ddpModel, ddpData, solverParams)

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
