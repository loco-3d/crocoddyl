import crocoddyL
import numpy as np
import pinocchio as se3
import os


display = True
plot = True

# Getting the robot model from the URDF file. Note that we use the URDF file
# inside our repository. By redefining the ROS_PACKAGE_PATH, Gepetto viewer is
# able to find the meshes
filename = str(os.path.dirname(os.path.abspath(__file__)))
os.environ['ROS_PACKAGE_PATH'] = filename
path = filename + '/hyq_description/'
urdf = path + 'robots/hyq_no_sensors.urdf'
robot = se3.robot_wrapper.RobotWrapper(urdf, path, se3.JointModelFreeFlyer())

# Create the contact information
contact_indices = [robot.model.getFrameId("lf_foot"),
                   robot.model.getFrameId("rf_foot"),
                   robot.model.getFrameId("rh_foot")]
contactPhase0 = crocoddyL.Phase(contact_indices, 0., np.inf)
contactInfo = crocoddyL.Multiphase([contactPhase0], 3)

# Create the dynamics and its integrator and discretizer
integrator = crocoddyL.EulerIntegrator()
discretizer = crocoddyL.EulerDiscretizer()
dynamics = crocoddyL.FloatingBaseMultibodyDynamics(integrator, discretizer,
                                              robot.model, contactInfo)

# Defining the SE3 task, the joint velocity and control regularizations
wSE3_term = 1e3 * np.ones((6,1))
wSE3_track = np.ones((6,1))
wv_reg = 1e-7 * np.vstack([ np.zeros((dynamics.nv(),1)),
                            np.ones((dynamics.nv(),1)) ])
wu_reg = 1e-7 * np.ones((dynamics.nu(),1))
se3_track = crocoddyL.SE3Cost(dynamics, wSE3_track)
se3_goal = crocoddyL.SE3Cost(dynamics, wSE3_term)
v_reg = crocoddyL.StateCost(dynamics, wv_reg)
u_reg = crocoddyL.ControlCost(dynamics, wu_reg)
# w_com = 1e3 * np.ones((3,1))
# com_cost = crocoddyL.CoMCost(dynamics, w_com)

# Adding the cost functions to the cost manager
costManager = crocoddyL.cost_manager.CostManager()
costManager.addTerminal(se3_goal, "se3_goal")
costManager.addRunning(se3_track, "se3_track")
costManager.addRunning(v_reg, "v_reg")
costManager.addRunning(u_reg, "u_reg")
#costManager.addRunning(com_cost)


# Setting up the DDP problem
timeline = np.arange(0.0, 0.25, 1e-3)  # np.linspace(0., 0.5, 51)
ddpModel = crocoddyL.DDPModel(dynamics, costManager)
ddpData = crocoddyL.DDPData(ddpModel, timeline)

# Setting up the initial conditions
q0 = robot.q0
q0[7:] = np.matrix([0., 0.75, -1.5,
                    0., -0.75, 1.5,
                    0., 0.75, -1.5,
                    0., -0.75, 1.5]).transpose()
x0 = np.vstack([q0, np.zeros((dynamics.nv(), 1))])
u0 = np.zeros((dynamics.nu(), 1))
U0 = [u0 for i in xrange(len(timeline)-1)]
ddpModel.setInitial(ddpData, xInit=x0, UInit=U0)

# Setting up the desired reference for each single cost function
# com_des = np.matrix([ [0.1], [0.], [0.] ])
frameRef = \
  crocoddyL.costs.SE3Task(se3.SE3(np.eye(3),
                     np.array([[0.1],[0.],[0.]])),
                     robot.model.getFrameId('base_link'))
Xref = [x0 for i in xrange(len(timeline))]
Uref = [u0 for i in xrange(len(timeline))]
Mref = [frameRef for i in xrange(len(timeline))]
# Cref = [com_des for i in xrange(len(timeline)-1)]
ddpModel.setTerminalReference(ddpData, Mref[-1], "se3_goal")
ddpModel.setRunningReference(ddpData, Mref[:-1], "se3_track")
ddpModel.setRunningReference(ddpData, Xref[:-1], "v_reg")


# Configuration the solver from YAML file and solving it
ddpParams = crocoddyL.DDPParams()
ddpParams.setFromConfigFile(filename + "/hyq_config.yaml")
crocoddyL.DDPSolver.solve(ddpModel, ddpData, ddpParams)


# Plotting the results
if plot:
  crocoddyL.plotDDPConvergence(ddpParams.cost_itr,
                               ddpParams.muLM_itr,
                               ddpParams.muV_itr,
                               ddpParams.gamma_itr,
                               ddpParams.theta_itr,
                               ddpParams.alpha_itr)


if display:
  T = timeline
  X = crocoddyL.DDPSolver.getStateTrajectory(ddpData)
  crocoddyL.visualizePlan(robot, T, x0, X)

# ddp.saveToFile('mu_1e2.txt')

# Printing the final goal
frame_idx = robot.model.getFrameId("base_link")
xf = ddpData.interval[-1].x
qf = xf[:dynamics.nq()]
print robot.framePlacement(qf, frame_idx)