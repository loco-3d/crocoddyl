import cddp
import numpy as np
import pinocchio as se3
import os

np.set_printoptions(linewidth=400, suppress=False, threshold=np.nan)

display = True
plot = True

filename = str(os.path.dirname(os.path.abspath(__file__)))

timeline = np.arange(0.0, 0.25, 1e-3)  # np.linspace(0., 0.5, 51)

# Create the integration and dynamics derivatives schemes. #TODO
ddpIntegrator = cddp.system.integrator.EulerIntegrator()
discretizer = cddp.system.discretizer.EulerDiscretizer()

# Create the ddp dynamics
ddpDynamics = cddp.dynamics.SpringMass(discretizer)


# Initial state
x0 = np.zeros((ddpDynamics.nx(),1))
u0 = np.zeros((ddpDynamics.nu(),1))


# Defining the velocity and control regularization
wx = 1e-7 * np.vstack([ np.zeros((ddpDynamics.nv(),1)), np.ones((ddpDynamics.nv(),1)) ])
wu = 1e-7 * np.ones((ddpDynamics.nu(),1))

#TODO: Why are we regularizing to zero posture!
x_cost = cddp.costs.multibody_dynamics.StateCost(ddpDynamics, wx)
xt_cost = cddp.costs.multibody_dynamics.StateCost(ddpDynamics, np.ones((2*ddpDynamics.nv(),1)))
u_cost = cddp.costs.multibody_dynamics.ControlCost(ddpDynamics, wu)

# Adding the cost functions to the cost manager
costManager = cddp.cost_manager.CostManager()
costManager.addRunning(x_cost, "x_cost")
costManager.addRunning(u_cost, "u_cost")
costManager.addTerminal(xt_cost, "xt_cost")

# Setting up the DDP problem
ddpModel = cddp.ddp_model.DDPModel(ddpDynamics, ddpIntegrator,
                                   costManager)
ddpData = cddp.ddp_model.DDPData(ddpModel, timeline)

# Setting the initial conditions
U0 = [u0 for i in xrange(len(timeline)-1)]
ddpModel.setInitial(ddpData, xInit=x0, UInit=U0)

# Setting the desired reference for each single cost function
xref = np.array([ [1.],[0.] ])
Xref = [xref for i in xrange(len(timeline))]
Uref = [u0 for i in xrange(len(timeline))]
ddpModel.setRunningReference(ddpData, Xref[:-1], "x_cost")
ddpModel.setTerminalReference(ddpData, Xref[-1], "xt_cost")

# Configuration the solver from YAML file
solverParams = cddp.solver.SolverParams()
solverParams.setFromConfigFile(filename + "/hyq_config.yaml")

# Solving the problem
cddp.Solver.solve(ddpModel, ddpData, solverParams)



if plot:
  X = cddp.Solver.getStateTrajectory(ddpData)
  U = cddp.Solver.getControlSequence(ddpData)
  cddp.utils.plotDDPSolution(X, U)

  cddp.utils.plotDDPConvergence(solverParams.cost_itr,
                                solverParams.muLM_itr,
                                solverParams.muV_itr, 
                                solverParams.gamma_itr,
                                solverParams.theta_itr,
                                solverParams.alpha_itr)


if display:
  T = timeline
  X = ddp.getStateTrajectory()
  cddp.visualizePlan(robot, T, x0, X)

# ddp.saveToFile('mu_1e2.txt')

# Printing the final goal
frame_idx = model.getFrameId(frame_name)
xf = ddp.intervals[-1].x
qf = xf[:model.nq]
print robot.framePosition(qf, frame_idx)