import crocoddyl
import numpy as np
import pinocchio as se3
import math
import os


display = True
plot = True


# Getting the robot model from the URDF file. Note that we use the URDF file
# installed by binary (through sudo-apt install robotpkg-talos-data)
filename = str(os.path.dirname(os.path.abspath(__file__)))
path = '/opt/openrobots/share/talos_data/'
urdf = path + 'robots/talos_reduced.urdf'
robot = se3.robot_wrapper.RobotWrapper(urdf, path, se3.JointModelFreeFlyer())

# Create the contact information
contact_indices = [robot.model.getFrameId("left_sole_link"),
                   robot.model.getFrameId("right_sole_link")]
contactPhase0 = crocoddyl.Phase(contact_indices, 0., np.inf)
contactInfo = crocoddyl.Multiphase([contactPhase0], 6)

# Create the dynamics and its integrator and discretizer
integrator = crocoddyl.EulerIntegrator()
discretizer = crocoddyl.EulerDiscretizer()
dynamics = crocoddyl.FloatingBaseMultibodyDynamics(integrator, discretizer,
                                                   robot.model, contactInfo)

# Defining the SE3 task, the joint velocity and control regularizations
wSE3_term = 1e3 * np.ones((6,1))
wSE3_track = 50. * np.ones((6,1))
wSO3_reg = 0.*1e-3 * np.ones((3,1))
q_ref = 1e-2 * np.array([0., 0., 0., 0., 0., 0.,                  # ff
                         1., 1., 1., 1., 1., 1.,                  # leg left
                         1., 1., 1., 1., 1., 1.,                  # leg right,
                         100., 100.,                              # torso
                         100., 100., 100., 5., 10., 10., 10., 1., # arm left
                         100., 100., 100., 5., 10., 10., 10., 1., # arm right
                         100., 100.,                              # head
                        ]).reshape((dynamics.nv(),1))
wv_reg = np.vstack([ q_ref, 1e-7 * np.ones((dynamics.nv(),1)) ])
wu_reg = 1e-7 * np.ones((dynamics.nu(),1))
se3_track = crocoddyl.SE3Cost(dynamics, wSE3_track)
se3_goal = crocoddyl.SE3Cost(dynamics, wSE3_term)
# so3_reg = crocoddyl.SO3Cost(dynamics, wSO3_reg)
v_reg = crocoddyl.StateCost(dynamics, wv_reg)
u_reg = crocoddyl.ControlCost(dynamics, wu_reg)
# w_com = 1e3 * np.ones((3,1))
# com_cost = crocoddyl.CoMCost(dynamics, w_com)

# Adding the cost functions to the cost manager
costManager = crocoddyl.CostManager()
costManager.addTerminal(se3_goal, "se3_goal")
costManager.addRunning(se3_track, "se3_track")
# costManager.addRunning(so3_reg, "so3_reg")
costManager.addRunning(v_reg, "v_reg")
costManager.addRunning(u_reg, "u_reg")
#costManager.addRunning(com_cost)


# Setting up the DDP problem
timeline = np.arange(0.0, 0.25, 1e-3)
ddpModel = crocoddyl.DDPModel(dynamics, costManager)
ddpData = crocoddyl.DDPData(ddpModel, timeline)

# Setting up the initial conditions
q0 = robot.q0
x0 = np.vstack([q0, np.zeros((dynamics.nv(), 1))])
u0 = np.zeros((dynamics.nu(), 1))
U0 = [u0 for i in xrange(len(timeline)-1)]
ddpModel.setInitial(ddpData, xInit=x0, UInit=U0)

# Setting up the desired reference for each single cost function
# com_des = np.matrix([ [0.1], [0.], [0.] ])
baseRef = crocoddyl.costs.SE3Task(
    se3.SE3(np.eye(3), np.array([[0.],[0.1],[0.]])),
    robot.model.getFrameId('base_link'))
# bodyRef = \
#   crocoddyl.costs.SO3Task(np.eye(3), robot.model.getFrameId('torso_1_link'))
Xref = [x0 for i in xrange(len(timeline))]
Uref = [u0 for i in xrange(len(timeline))]
oMb_ref = []
t = 0.
for i in xrange(len(timeline)):
  t += 1e-3
  p = np.array([ [0.],[0.1 * math.sin(2 * math.pi * t * 4)],[0.] ])
  M = crocoddyl.costs.SE3Task(
    se3.SE3(np.eye(3), p),
    robot.model.getFrameId('base_link'))
  oMb_ref.append(M)
# oMb_ref = [baseRef for i in xrange(len(timeline))]
# oMt_ref = [bodyRef for i in xrange(len(timeline))]
# Cref = [com_des for i in xrange(len(timeline)-1)]
ddpModel.setTerminalReference(ddpData, oMb_ref[-1], "se3_goal")
ddpModel.setRunningReference(ddpData, oMb_ref[:-1], "se3_track")
# ddpModel.setRunningReference(ddpData, oMt_ref[:-1], "so3_reg")
ddpModel.setRunningReference(ddpData, Xref[:-1], "v_reg")


# Configuration the solver from YAML file and solving it
ddpParams = crocoddyl.DDPParams()
ddpParams.setFromConfigFile(filename + "/talos_config.yaml")
crocoddyl.DDPSolver.solve(ddpModel, ddpData, ddpParams)


# Plotting the results
if plot:
  crocoddyl.plotDDPConvergence(
    crocoddyl.DDPSolver.getCostSequence(ddpData),
    crocoddyl.DDPSolver.getLMRegularizationSequence(ddpData),
    crocoddyl.DDPSolver.getVRegularizationSequence(ddpData),
    crocoddyl.DDPSolver.getGammaSequence(ddpData),
    crocoddyl.DDPSolver.getThetaSequence(ddpData),
    crocoddyl.DDPSolver.getAlphaSequence(ddpData))


if display:
  T = timeline
  X = crocoddyl.DDPSolver.getStateTrajectory(ddpData)
  crocoddyl.visualizePlan(robot, T, x0, X)

# ddp.saveToFile('mu_1e2.txt')

# Printing the final goal
frame_idx = robot.model.getFrameId("base_link")
xf = ddpData.interval[-1].x
qf = xf[:dynamics.nq()]
print robot.framePlacement(qf, frame_idx)