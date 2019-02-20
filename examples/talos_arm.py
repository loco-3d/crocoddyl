from crocoddyl import *
import pinocchio
import numpy as np
import sys

WITHDISPLAY =  'disp' in sys.argv
WITHPLOT = 'plot' in sys.argv

# In this example test, we will solve the reaching-goal task with the Talos arm.
# For that, we use the forward dynamics (with its analytical derivatives)
# developed inside crocoddyl; it describes inside DifferentialActionModelFullyActuated class.
# Finally, we use an Euler sympletic integration scheme.


# First, let's load the Pinocchio model for the Talos arm. And then, let's
# create a cost model per the running and terminal action model.
robot = loadTalosArm()
runningCostModel = CostModelSum(robot.model)
terminalCostModel = CostModelSum(robot.model)

# Note that we need to include a cost model (i.e. set of cost functions) in
# order to fully define the action model for our optimal control problem.
# For this particular example, we formulate three running-cost functions: 
# goal-tracking cost, state and control regularization; and one terminal-cost:
# goal cost. First, let's create the common cost functions.
frameName = 'gripper_left_joint'
state = StatePinocchio(robot.model)
SE3ref = pinocchio.SE3(np.eye(3), np.array([ [.0],[.0],[.4] ]))
goalTrackingCost = CostModelFramePlacement(robot.model,
                                       nu=robot.model.nv,
                                       frame=robot.model.getFrameId(frameName),
                                       ref=SE3ref)
xRegCost = CostModelState(robot.model,
                          state,
                          ref=state.zero(),
                          nu=robot.model.nv)
uRegCost = CostModelControl(robot.model,nu=robot.model.nv)

# Then let's added the running and terminal cost functions
runningCostModel.addCost( name="pos", weight = 1e-3, cost = goalTrackingCost)
runningCostModel.addCost( name="regx", weight = 1e-7, cost = xRegCost) 
runningCostModel.addCost( name="regu", weight = 1e-7, cost = uRegCost)
terminalCostModel.addCost( name="pos", weight = 1, cost = goalTrackingCost)


# Next, we need to create an action model for running and terminal knots. The
# forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFullyActuated.
runningModel = IntegratedActionModelEuler(
    DifferentialActionModelFullyActuated(robot.model, runningCostModel))
terminalModel = IntegratedActionModelEuler(
    DifferentialActionModelFullyActuated(robot.model, terminalCostModel))

# Defining the time duration for running action models and the terminal one
dt = 1e-3
runningModel.timeStep = dt


# For this optimal control problem, we define 250 knots (or running action
# models) plus a terminal knot
T = 250
q0 = [0.173046, 1., -0.52366, 0., 0., 0.1, -0.005]
x0 = np.hstack([q0, np.zeros(robot.model.nv)])
problem = ShootingProblem(x0, [ runningModel ]*T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = SolverDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
ddp.callback = [ CallbackDDPVerbose() ]
if WITHPLOT:       ddp.callback.append(CallbackDDPLogger())
if WITHDISPLAY:    ddp.callback.append(CallbackSolverDisplay(talos_legs,4,1,cameraTF))

# Solving it with the DDP algorithm
ddp.solve()

# Plotting the solution and the DDP convergence
if WITHPLOT:
    log = ddp.callback[0]
    plotOCSolution(log.xs, log.us)
    plotDDPConvergence(log.costs,log.control_regs,
                       log.state_regs,log.gm_stops,
                       log.th_stops,log.steps)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY: CallbackSolverDisplay(robot)(ddp)

# Printing the reached position
frame_idx = robot.model.getFrameId(frameName)
print
print "The reached pose by the wrist is"
print ddp.datas()[-1].differential.pinocchio.oMf[frame_idx]

