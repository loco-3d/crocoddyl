from crocoddyl import StatePinocchio
from crocoddyl import DifferentialActionModel, IntegratedActionModelEuler
from crocoddyl import CostModelPosition, CostModelPosition6D
from crocoddyl import CostModelState, CostModelControl
from crocoddyl import ShootingProblem, SolverDDP
from crocoddyl import SolverLogger, SolverPrinter, SolverDisplay
from crocoddyl import loadTalosArm
from crocoddyl import plotOCSolution, plotDDPConvergence
import pinocchio
import numpy as np



# In this example test, we will solve the reaching-goal task with the Talos arm.
# For that, we use the forward dynamics (with its analytical derivatives)
# developed inside crocoddyl; it describes inside DifferentialActionModel class.
# Finally, we use an Euler sympletic integration scheme.


# First, let's load the Pinocchio model for the Talos arm. And then, let's
# create an action model for running and terminal knots. The forward dynamics
# (computed using ABA) are implemented inside DifferentialActionModel.
robot = loadTalosArm()
runningModel = IntegratedActionModelEuler(DifferentialActionModel(robot.model))
terminalModel = IntegratedActionModelEuler(DifferentialActionModel(robot.model))

# Defining the time duration for running action models and the terminal one
dt = 1e-3
runningModel.timeStep = dt

# Note that we need to include a cost model (i.e. set of cost functions) in
# order to fully define the action model for our optimal control problem.
# For this particular example, we formulate three running-cost functions: 
# goal-tracking cost, state and control regularization; and one terminal-cost:
# goal cost. First, let's create the common cost functions.
frameName = 'gripper_left_joint' #gripper_left_fingertip_2_link'
state = StatePinocchio(robot.model)
SE3ref = pinocchio.SE3(np.eye(3), np.array([ [.0],[.0],[.4] ]))
goalTrackingCost = CostModelPosition6D(robot.model,
                                       nu=robot.model.nv,
                                       frame=robot.model.getFrameId(frameName),
                                       ref=SE3ref)
# goalTrackingCost = CostModelPosition(robot.model,
#                                      nu=robot.model.nv,
#                                      frame=robot.model.getFrameId(frameName),
#                                      ref=np.array([.0,.0,.4]))
xRegCost = CostModelState(robot.model,
                          state,
                          ref=state.zero(),
                          nu=robot.model.nv)
uRegCost = CostModelControl(robot.model,nu=robot.model.nv)

# Then let's added the running and terminal cost functions
runningCostModel = runningModel.differential.costs
runningCostModel.addCost( name="pos", weight = 1e-3, cost = goalTrackingCost)
runningCostModel.addCost( name="regx", weight = 1e-7, cost = xRegCost) 
runningCostModel.addCost( name="regu", weight = 1e-7, cost = uRegCost)
terminalCostModel = terminalModel.differential.costs
terminalCostModel.addCost( name="pos", weight = 1, cost = goalTrackingCost)


# For this optimal control problem, we define 250 knots (or running action
# models) plus a terminal knot
T = 250
q0 = [0.173046, 1., -0.52366, 0., 0., 0.1, -0.005]
x0 = np.hstack([q0, np.zeros(robot.model.nv)])
problem = ShootingProblem(x0, [ runningModel ]*T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = SolverDDP(problem)
ddp.callback = [SolverLogger(), SolverPrinter(1), SolverDisplay(robot,4)]

# Solving it with the DDP algorithm
ddp.solve()


# Plotting the solution and the DDP convergence
log = ddp.callback[0]
plotOCSolution(log.xs, log.us)
plotDDPConvergence(log.costs,log.control_regs,
                   log.state_regs,log.gm_stops,
                   log.th_stops,log.steps)


# Visualizing the solution in gepetto-viewer
SolverDisplay(robot)(ddp)

# Printing the reached position
frame_idx = robot.model.getFrameId(frameName)
xT = log.xs[-1]
qT = np.asmatrix(xT[:robot.model.nq]).T
print
print "The reached pose by the wrist is"
print robot.framePlacement(qT, frame_idx)