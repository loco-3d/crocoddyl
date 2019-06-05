import sys

import numpy as np
import pinocchio
from crocoddyl import (CallbackDDPLogger, CallbackDDPVerbose, CallbackSolverDisplay, CostModelControl,
                       CostModelFramePlacement, CostModelState, CostModelSum, DifferentialActionModelFullyActuated,
                       IntegratedActionModelEuler, ShootingProblem, SolverDDP, StatePinocchio, loadTalosArm,
                       plotDDPConvergence, plotOCSolution)

WITHDISPLAY = 'disp' in sys.argv
WITHPLOT = 'plot' in sys.argv

# In this example test, we will solve the reaching-goal task with the Talos arm.
# For that, we use the forward dynamics (with its analytical derivatives)
# developed inside crocoddyl; it describes inside DifferentialActionModelFullyActuated class.
# Finally, we use an Euler sympletic integration scheme.

# First, let's load the Pinocchio model for the Talos arm.
robot = loadTalosArm()
rmodel = robot.model
rdata = rmodel.createData()

# Create a cost model per the running and terminal action model.
runningCostModel = CostModelSum(rmodel)
terminalCostModel = CostModelSum(rmodel)

# Note that we need to include a cost model (i.e. set of cost functions) in
# order to fully define the action model for our optimal control problem.
# For this particular example, we formulate three running-cost functions:
# goal-tracking cost, state and control regularization; and one terminal-cost:
# goal cost. First, let's create the common cost functions.
frameName = 'gripper_left_joint'
state = StatePinocchio(rmodel)
SE3ref = pinocchio.SE3(np.eye(3), np.array([[.0], [.0], [.4]]))
goalTrackingCost = CostModelFramePlacement(rmodel, nu=rmodel.nv, frame=rmodel.getFrameId(frameName), ref=SE3ref)
xRegCost = CostModelState(rmodel, state, ref=state.zero(), nu=rmodel.nv)
uRegCost = CostModelControl(rmodel, nu=rmodel.nv)

# Then let's added the running and terminal cost functions
runningCostModel.addCost(name="pos", weight=1e-3, cost=goalTrackingCost)
runningCostModel.addCost(name="regx", weight=1e-7, cost=xRegCost)
runningCostModel.addCost(name="regu", weight=1e-7, cost=uRegCost)
terminalCostModel.addCost(name="pos", weight=1, cost=goalTrackingCost)

# Next, we need to create an action model for running and terminal knots. The
# forward dynamics (computed using ABA) are implemented
# inside DifferentialActionModelFullyActuated.
runningModel = IntegratedActionModelEuler(DifferentialActionModelFullyActuated(rmodel, runningCostModel))
terminalModel = IntegratedActionModelEuler(DifferentialActionModelFullyActuated(rmodel, terminalCostModel))

# Defining the time duration for running action models and the terminal one
dt = 1e-3
runningModel.timeStep = dt

# For this optimal control problem, we define 250 knots (or running action
# models) plus a terminal knot
T = 250
q0 = [0.173046, 1., -0.52366, 0., 0., 0.1, -0.005]
x0 = np.hstack([q0, np.zeros(rmodel.nv)])
problem = ShootingProblem(x0, [runningModel] * T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = SolverDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
ddp.callback = [CallbackDDPVerbose()]
if WITHPLOT:
    ddp.callback.append(CallbackDDPLogger())
if WITHDISPLAY:
    ddp.callback.append(CallbackSolverDisplay(robot, 4, 1, cameraTF))

# Solving it with the DDP algorithm
ddp.solve()

# Plotting the solution and the DDP convergence
if WITHPLOT:
    log = ddp.callback[1]
    plotOCSolution(log.xs, log.us, figIndex=1, show=False)
    plotDDPConvergence(log.costs, log.control_regs, log.state_regs, log.gm_stops, log.th_stops, log.steps, figIndex=2)

# Visualizing the solution in gepetto-viewer
if WITHDISPLAY:
    from crocoddyl.diagnostic import displayTrajectory
    displayTrajectory(robot, ddp.xs, runningModel.timeStep)

# Printing the reached position
frame_idx = rmodel.getFrameId(frameName)
print
print("The reached pose by the wrist is")
print(ddp.datas()[-1].differential.pinocchio.oMf[frame_idx])
