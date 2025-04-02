import numpy as np
from unicycle_utils import plotUnicycleSolution

import crocoddyl

# Creating an action model for the unicycle system
model = crocoddyl.ActionModelUnicycle()

# Setting up the cost weights
model.r = [10.0, 1.0]  # state weight  # control weight

# Formulating the optimal control problem
T = 20  # number of knots
x0 = np.matrix([-1.0, -1.0, 1.0]).T  # x,y,theta
problem = crocoddyl.ShootingProblem(x0, [model] * T, model)

# Creating the DDP solver for this OC problem, defining a logger
ddp = crocoddyl.SolverDDP(problem)
ddp.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

# Solving it with the DDP algorithm
ddp.solve()

# Plotting the solution, solver convergence and unicycle motion
log = ddp.getCallbacks()[0]
crocoddyl.plotOCSolution(log.xs, log.us, figIndex=1, show=False)
crocoddyl.plotConvergence(
    log.costs,
    log.pregs,
    log.dregs,
    log.grads,
    log.stops,
    log.steps,
    figIndex=2,
    show=False,
)
plotUnicycleSolution(log.xs, figIndex=3, show=True)
