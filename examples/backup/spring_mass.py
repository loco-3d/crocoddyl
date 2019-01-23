import crocoddyl
import numpy as np
import pinocchio as se3
import os


plot = True

# Create the dynamics and its integrator and discretizer
integrator = crocoddyl.EulerIntegrator()
discretizer = crocoddyl.EulerDiscretizer()
dynamics = crocoddyl.SpringMass(integrator, discretizer)

# Defining the cost functions for goal state, state tracking, and control
# regularization
wx_term = 1e3 * np.ones((2 * dynamics.nv(),1))
wx_track = 10. * np.ones((2 * dynamics.nv(),1))
wu_reg = 1e-2 * np.ones((dynamics.nu(),1))
x_track = crocoddyl.StateCost(dynamics, wx_track)
xT_goal = crocoddyl.StateCost(dynamics, wx_term)
u_reg = crocoddyl.ControlCost(dynamics, wu_reg)

# Adding the cost functions to the cost manager
costManager = crocoddyl.CostManager()
costManager.addTerminal(xT_goal, "xT_goal")
costManager.addRunning(x_track, "x_track")
costManager.addRunning(u_reg, "u_reg")


# Setting up the DDP problem
timeline = np.arange(0.0, 0.25, 1e-3)
ddpModel = crocoddyl.DDPModel(dynamics, costManager)
ddpData = crocoddyl.DDPData(ddpModel, timeline)

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
ddpParams = crocoddyl.DDPParams()
ddpParams.setFromConfigFile(filename + "/spring_mass_config.yaml")
crocoddyl.DDPSolver.solve(ddpModel, ddpData, ddpParams)


# Plotting the results
if plot:
  X = crocoddyl.DDPSolver.getStateTrajectory(ddpData)
  U = crocoddyl.DDPSolver.getControlSequence(ddpData)
  crocoddyl.plotDDPSolution(X, U)

  crocoddyl.plotDDPConvergence(
    crocoddyl.DDPSolver.getCostSequence(ddpData),
    crocoddyl.DDPSolver.getLMRegularizationSequence(ddpData),
    crocoddyl.DDPSolver.getVRegularizationSequence(ddpData),
    crocoddyl.DDPSolver.getGammaSequence(ddpData),
    crocoddyl.DDPSolver.getThetaSequence(ddpData),
    crocoddyl.DDPSolver.getAlphaSequence(ddpData))