
from crocoddyl import SolverDDP, CallbackDDPLogger, CallbackDDPVerbose
from crocoddyl import ShootingProblem
from crocoddyl import ActionModelUnicycleVar
from crocoddyl import plotOCSolution
import numpy as np


# Creating an action model for the unicycle system
model = ActionModelUnicycleVar()


# Formulating the optimal control problem
T = 10 # number of knots
x0 = np.array([ -1, -2, 1, 2 ]) # initial state
problem = ShootingProblem(x0, [ model ]*T, model )


# Creating the DDP solver for this OC problem, defining a logger
ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPLogger(), CallbackDDPVerbose(1)]

# Solving it with the DDP algorithm
ddp.solve()


# Plotting the solution
log = ddp.callback[0]
plotOCSolution(log.xs, log.us)
