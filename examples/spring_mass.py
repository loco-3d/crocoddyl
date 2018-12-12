import crocoddyL
import numpy as np
import pinocchio as se3
import os


plot = True

# Create the dynamics and its integrator and discretizer
integrator = crocoddyL.EulerIntegrator()
discretizer = crocoddyL.EulerDiscretizer()
dynamics = crocoddyL.SpringMass(integrator, discretizer)

# Defining the cost functions for goal state, state tracking, and control
# regularization
wx_term = 1e3 * np.ones((2 * dynamics.nv(),1))
wx_track = 10. * np.ones((2 * dynamics.nv(),1))
wu_reg = 1e-2 * np.ones((dynamics.nu(),1))
x_track = crocoddyL.StateCost(dynamics, wx_track)
xT_goal = crocoddyL.StateCost(dynamics, wx_term)
u_reg = crocoddyL.ControlCost(dynamics, wu_reg)

# Adding the cost functions to the cost manager
costManager = crocoddyL.CostManager()
costManager.addTerminal(xT_goal, "xT_goal")
costManager.addRunning(x_track, "x_track")
costManager.addRunning(u_reg, "u_reg")


# Setting up the DDP problem
timeline = np.arange(0.0, 0.25, 1e-3)
ddpModel = crocoddyL.DDPModel(dynamics, costManager)
ddpData = crocoddyL.DDPData(ddpModel, timeline)

# Setting up the initial conditions
x0 = np.zeros((dynamics.nx(),1))
u0 = np.zeros((dynamics.nu(),1))
U0 = [u0 for i in xrange(len(timeline)-1)]
ddpModel.setInitial(ddpData, xInit=x0, UInit=U0)

# Setting up the desired reference for each single cost function
xref = np.array([ [1.],[0.] ])
Xref = [xref for i in xrange(len(timeline))]
ddpModel.setRunningReference(ddpData, Xref[:-1], "x_track")
ddpModel.setTerminalReference(ddpData, Xref[-1], "xT_goal")


# Configuration the solver from YAML file and solving it
filename = str(os.path.dirname(os.path.abspath(__file__)))
solverParams = crocoddyL.SolverParams()
solverParams.setFromConfigFile(filename + "/spring_mass_config.yaml")
crocoddyL.Solver.solve(ddpModel, ddpData, solverParams)


# Plotting the results
if plot:
  X = crocoddyL.Solver.getStateTrajectory(ddpData)
  U = crocoddyL.Solver.getControlSequence(ddpData)
  crocoddyL.plotDDPSolution(X, U)

  crocoddyL.plotDDPConvergence(solverParams.cost_itr,
                          solverParams.muLM_itr,
                          solverParams.muV_itr, 
                          solverParams.gamma_itr,
                          solverParams.theta_itr,
                          solverParams.alpha_itr)