import numpy as np

import crocoddyl
from unicycle_utils import plotUnicycleSolution

# Creating an action model for the unicycle system
model = crocoddyl.ActionModelUnicycle()

# Setting up the cost weights
model.r = [
    10.,  # state weight
    1.  # control weight
]

# Formulating the optimal control problem
T = 20  # number of knots
x0 = np.matrix([-1., -1., 1.]).T  #x,y,theta
problem = crocoddyl.ShootingProblem(x0, [model] * T, model)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

# Solving it with the DDP algorithm
ddp.solve()

# Plotting the solution, solver convergence and unicycle motion
log = ddp.getCallbacks()[0]
crocoddyl.plotOCSolution(log.xs, log.us)
crocoddyl.plotConvergence(log.costs, log.control_regs, log.state_regs, log.gm_stops, log.th_stops, log.steps)
plotUnicycleSolution(log.xs)
