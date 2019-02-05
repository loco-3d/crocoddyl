from crocoddyl import *
import pinocchio
import numpy as np


robot = loadTalosArm()
robot.q0.flat[:] = [  2,1.5,-2,0,0,0,0 ]
robot.model.armature[:] = .2
frameId = robot.model.getFrameId('gripper_left_joint')
DT = 1e-2
T  = 15

State = StatePinocchio(robot.model)

ps = [
    np.array([ 0.4,0  ,.4 ]),
    np.array([ 0.4,0.4,.4 ]),
    np.array([ 0.4,0.4, 0 ]),
    np.array([ 0.4,0,   0 ]),
    ]

colors = [
    [ 1,0,0, 1],
    [ 0,1,0, 1],
    [ 0,0,1, 1],
    [ 1,0,1, 1],
    ]

robot.initDisplay(loadModel=True)
gv = robot.viewer.gui
for i,p in enumerate(ps):
    gv.addSphere('world/point%d'%i,.1,colors[i])
    gv.applyConfiguration('world/point%d'%i, p.tolist()+[0,0,0,1] )
gv.refresh()


models     = [ DifferentialActionModelManipulator(robot.model) for p in ps]
termmodels = [ DifferentialActionModelManipulator(robot.model) for p in ps]

costTrack = [ CostModelFrameTranslation(robot.model,frame=frameId,ref=p) for p in ps ]
costXReg = CostModelState(robot.model,
                          StatePinocchio(robot.model))
costUReg = CostModelControl(robot.model,nu=robot.model.nv)

# Then let's added the running and terminal cost functions
for model,cost in zip(models,costTrack):
    model.costs.addCost( name="pos", weight = 1, cost = cost)
    model.costs.addCost( name="xreg", weight = 1e-4, cost = costXReg)
    model.costs.addCost( name="ureg", weight = 1e-7, cost = costUReg)
for model,cost in zip(termmodels,costTrack):
    model.costs.addCost( name="pos", weight = 1000, cost = cost)
    model.costs.addCost( name="xreg", weight = 1e-4, cost = costXReg)
    model.costs.addCost( name="ureg", weight = 1e-7, cost = costUReg)

x0 = np.concatenate([ m2a(robot.q0), np.zeros(robot.model.nv)])
seqs = [  [ IntegratedActionModelEuler(model) ]*T+[IntegratedActionModelEuler(termmodel)]
          for model,termmodel in zip(models,termmodels) ]
problem = ShootingProblem(x0, sum(seqs,[])[:-1], seqs[-1][-1])

# Creating the DDP solver for this OC problem, defining a logger
ddp = SolverDDP(problem)
cameraTF = [2., 2.68, 0.54, 0.2, 0.62, 0.72, 0.22]
ddp.callback = [CallbackDDPVerbose() ]

# Solving it with the DDP algorithm
ddp.solve()

# Penalty if you want.
for i in range(4,6):
    for m in termmodels: m.costs['pos'].weight = 10**i
    ddp.solve(init_xs=ddp.xs,init_us=ddp.us,maxiter=10)

# Visualizing the solution in gepetto-viewer
CallbackSolverDisplay(robot)(ddp)

for i,s in enumerate(seqs+[1]):
    print ddp.datas()[i*T].differential.costs['pos'].pinocchio.oMf[frameId].translation.T
