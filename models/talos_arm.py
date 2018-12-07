import cddp
import numpy as np
import pinocchio as se3
import os


display = True
plot = True


# Getting the robot model from the URDF file. Note that we use the URDF file
# installed by binary (through sudo-apt install robotpkg-talos-data)
filename = str(os.path.dirname(os.path.abspath(__file__)))
path = '/opt/openrobots/share/talos_data/'
urdf = path + 'robots/talos_left_arm.urdf'
robot = se3.robot_wrapper.RobotWrapper(urdf, path)
model = robot.model

# Create the dynamics and its integrator and discretizer
integrator = cddp.EulerIntegrator()
discretizer = cddp.EulerDiscretizer()
dynamics = cddp.ForwardDynamics(integrator, discretizer, robot.model)

# Defining the SE3 task, the joint velocity and control regularization
wSE3_goal = np.ones((6,1))
wSE3_track = 1e-3 * np.ones((6,1))
wv_reg = 1e-7 * np.vstack([ np.zeros((dynamics.nv(),1)),
                            np.ones((dynamics.nv(),1)) ])
wu_reg = 1e-7 * np.ones((dynamics.nu(),1))
se3_goal = cddp.SE3Cost(dynamics, wSE3_goal)
se3_track = cddp.SE3Cost(dynamics, wSE3_track)
v_reg = cddp.StateCost(dynamics, wv_reg)
u_reg = cddp.ControlCost(dynamics, wu_reg)

# Adding the cost functions to the cost manager
costManager = cddp.CostManager()
costManager.addTerminal(se3_goal, "se3_goal")
costManager.addRunning(se3_track, "se3_track")
costManager.addRunning(v_reg, "v_reg")
costManager.addRunning(u_reg, "u_reg")


# Setting up the DDP problem
timeline = np.arange(0.0, 0.25, 1e-3)  # np.linspace(0., 0.5, 51)
ddpModel = cddp.DDPModel(dynamics, costManager)
ddpData = cddp.DDPData(ddpModel, timeline)

# Setting up the initial conditions
q0 = np.matrix([0.173046, 1., -0.525366, 0., 0., 0.1, -0.005]).T
x0 = np.vstack([q0, np.zeros((dynamics.nv(), 1))])
u0 = np.zeros((dynamics.nv(), 1))
U0 = [u0 for i in xrange(len(timeline)-1)]
ddpModel.setInitial(ddpData, xInit=x0, UInit=U0)


# Setting up the desired reference for each single cost function
frameRef = \
  cddp.costs.SE3Task(se3.SE3(np.eye(3),
                     np.array([[0.],[0.],[0.4]])),
                     model.getFrameId('gripper_left_joint'))
Xref = [x0 for i in xrange(len(timeline))]
Uref = [u0 for i in xrange(len(timeline))]
Mref = [frameRef for i in xrange(len(timeline))]
ddpModel.setTerminalReference(ddpData, Mref[-1], "se3_goal")
ddpModel.setRunningReference(ddpData, Mref[:-1], "se3_track")
ddpModel.setRunningReference(ddpData, Xref[:-1], "v_reg")



# Configuration the solver from YAML file and solving it
solverParams = cddp.SolverParams()
solverParams.setFromConfigFile(filename + "/talos_arm_config.yaml")
cddp.Solver.solve(ddpModel, ddpData, solverParams)


# Plotting the results
if plot:
  cddp.plotDDPConvergence(solverParams.cost_itr,
                          solverParams.muLM_itr,
                          solverParams.muV_itr, 
                          solverParams.gamma_itr,
                          solverParams.theta_itr,
                          solverParams.alpha_itr)


if display:
  T = timeline
  X = cddp.Solver.getStateTrajectory(ddpData)
  cddp.visualizePlan(robot, T, x0, X)


# Printing the final goal
frame_idx = robot.model.getFrameId("gripper_left_joint")
xf = ddpData.intervalDataVector[-1].x
qf = xf[:dynamics.nq()]
print robot.framePlacement(qf, frame_idx)