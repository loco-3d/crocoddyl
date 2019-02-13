from crocoddyl import *
import numpy as np


# Creating an action model for the unicycle system
model = ActionModelUnicycle()

# Setting up the cost weights
model.costWeights = [
    10.,   # state weight
    1.  # control weight
]


# Formulating the optimal control problem
T = 20 # number of knots
x0 = np.array([ -1., -1., 1. ]) #x,y,theta
problem = ShootingProblem(x0, [ model ]*T, model )


# Creating the DDP solver for this OC problem, defining a logger
ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPLogger(), CallbackDDPVerbose(1)]

# Solving it with the DDP algorithm
ddp.solve()


# Plotting the solution, solver convergence and unycicle motion
log = ddp.callback[0]
plotOCSolution(log.xs, log.us)
plotDDPConvergence(log.costs,log.control_regs,
                   log.state_regs,log.gm_stops,
                   log.th_stops,log.steps)
plotUnicycleSolution(log.xs)