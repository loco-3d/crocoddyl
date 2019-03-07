import numpy as np

import pinocchio
from crocoddyl import *

robot = loadTalosArm()
robot.q0.flat[:] = [2, 1.5, -2, 0, 0, 0, 0]
robot.model.armature[:] = .2
frameId = robot.model.getFrameId('gripper_left_joint')
DT = 1e-2
T = 15

State = StatePinocchio(robot.model)

ps = [
    np.array([0.4, 0, .4]),
    np.array([0.4, 0.4, .4]),
    np.array([0.4, 0.4, 0]),
    np.array([0.4, 0, 0]),
]

colors = [
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 1, 1],
]

robot.initDisplay(loadModel=True)
gv = robot.viewer.gui
for i, p in enumerate(ps):
    gv.addSphere('world/point%d' % i, .1, colors[i])
    gv.applyConfiguration('world/point%d' % i, p.tolist() + [0, 0, 0, 1])
gv.refresh()

# State and control regularization costs
costXReg = CostModelState(robot.model, StatePinocchio(robot.model))
costUReg = CostModelControl(robot.model, nu=robot.model.nv)

# Then let's added the running and terminal cost functions per each action
# model
runningModels = []
terminalModels = []
for p in ps:
    # Create the tracking cost
    costTrack = CostModelFrameTranslation(robot.model, frame=frameId, ref=p)

    # Create the running action model
    runningCostModel = CostModelSum(robot.model)
    runningCostModel.addCost(name="pos", weight=1, cost=costTrack)
    runningCostModel.addCost(name="xreg", weight=1e-4, cost=costXReg)
    runningCostModel.addCost(name="ureg", weight=1e-7, cost=costUReg)
    runningModels += [DifferentialActionModelFullyActuated(robot.model, runningCostModel)]

    # Create the terminal action model
    terminalCostModel = CostModelSum(robot.model)
    terminalCostModel.addCost(name="pos", weight=1000, cost=costTrack)
    terminalCostModel.addCost(name="xreg", weight=1e-4, cost=costXReg)
    terminalCostModel.addCost(name="ureg", weight=1e-7, cost=costUReg)
    terminalModels += [DifferentialActionModelFullyActuated(robot.model, terminalCostModel)]

x0 = np.concatenate([m2a(robot.q0), np.zeros(robot.model.nv)])
seqs = [[IntegratedActionModelEuler(runningModel)] * T + [IntegratedActionModelEuler(terminalModel)]
        for runningModel, terminalModel in zip(runningModels, terminalModels)]
problem = ShootingProblem(x0, sum(seqs, [])[:-1], seqs[-1][-1])

# Creating the DDP solver for this OC problem, defining a logger
ddp = SolverDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
ddp.callback = [CallbackDDPVerbose()]

# Solving it with the DDP algorithm
ddp.solve()

# Penalty if you want.
for i in range(4, 6):
    for m in terminalModels:
        m.costs['pos'].weight = 10**i
    ddp.solve(init_xs=ddp.xs, init_us=ddp.us, maxiter=10)

# Visualizing the solution in gepetto-viewer
CallbackSolverDisplay(robot)(ddp)

for i, s in enumerate(seqs + [1]):
    print ddp.datas()[i * T].differential.costs['pos'].pinocchio.oMf[frameId].translation.T
