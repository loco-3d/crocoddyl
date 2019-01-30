from crocoddyl import StatePinocchio
from crocoddyl import DifferentialActionModel, IntegratedActionModelEuler
from crocoddyl import CostModelFrameTranslation, CostModelFramePlacement
from crocoddyl import CostModelState, CostModelControl
from crocoddyl import ShootingProblem, SolverDDP
from crocoddyl import CallbackDDPLogger, CallbackDDPVerbose, CallbackSolverDisplay
from crocoddyl import loadTalosArm
from crocoddyl import plotOCSolution, plotDDPConvergence
import pinocchio
import numpy as np


robot = loadTalosArm()
frameId = robot.model.getFrameId('gripper_left_joint')
DT = 2e-3
T  = 50

runningModel = IntegratedActionModelEuler(DifferentialActionModel(robot.model))
runningModel.timeStep = DT
terminalModel = IntegratedActionModelEuler(DifferentialActionModel(robot.model))
terminalModel.timeStep = DT

state = StatePinocchio(robot.model)
SE3ref = pinocchio.SE3(np.eye(3), np.array([ [.0],[.0],[.4] ]))
goalTrackingCost = CostModelFramePlacement(robot.model,
                                           frame=frameId,
                                           ref=SE3ref)
# goalTrackingCost = CostModelFrameTranslation(robot.model,
#                                        nu=robot.model.nv,
#                                              frame=frameId,
#                                              ref=np.array([0,0,.4]))
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


q0 = [0.173046, 1., -0.52366, 0., 0., 0.1, -0.005]
x0 = np.hstack([q0, np.zeros(robot.model.nv)])
problem = ShootingProblem(x0, [ runningModel ]*T, terminalModel)

# Creating the DDP solver for this OC problem, defining a logger
ddp = SolverDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
ddp.callback = [CallbackDDPLogger(), CallbackDDPVerbose(1), CallbackSolverDisplay(robot,4,cameraTF)]

# Solving it with the DDP algorithm
ddp.solve()

# Visualizing the solution in gepetto-viewer
CallbackSolverDisplay(robot)(ddp)


for i in range(5):
    terminalCostModel['pos'].weight = 10*1
    ddp.solve(init_xs=ddp.xs,init_us=ddp.us,maxiter=10)


# Printing the reached position
xT = ddp.xs[-1]
qT = np.asmatrix(xT[:robot.model.nq]).T
print
print "The reached pose by the wrist is"
print robot.framePlacement(qT, frameId)
