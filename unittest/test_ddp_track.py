import refact
import pinocchio
from pinocchio.utils import *
from numpy.linalg import inv,norm,pinv
from numpy import dot,asarray
from continuous import IntegratedActionModelEuler, DifferentialActionModelNumDiff,StatePinocchio,CostModelSum,CostModelPinocchio,CostModelPosition,CostModelState,CostModelControl,DifferentialActionModel,CostModelPlacementVelocity
from contact import ContactModel6D,ActuationModelFreeFloating,DifferentialActionModelFloatingInContact,ContactModelMultiple
import warnings
from numpy.linalg import inv,pinv,norm,svd,eig
from crocoddyl import ActivationModelWeightedQuad
from robots import loadTalosLegs

m2a = lambda m: np.array(m.flat)
a2m = lambda a: np.matrix(a).T
absmax = lambda A: np.max(abs(A))
absmin = lambda A: np.min(abs(A))

robot = loadTalosLegs()

opPointName = 'right_sole_link'
contactName = 'left_sole_link'

opPointName,contactName = contactName,opPointName


rmodel = robot.model
rmodel.armature = np.matrix([ 0 ]*robot.model.nv).T
for j in robot.model.joints[1:]:
    if j.shortname()!='JointModelFreeFlyer':
        rmodel.armature[j.idx_v:j.idx_v+j.nv]=1
                
qmin = rmodel.lowerPositionLimit; qmin[:7]=-1; rmodel.lowerPositionLimit = qmin
qmax = rmodel.upperPositionLimit; qmax[:7]= 1; rmodel.upperPositionLimit = qmax
State = StatePinocchio(rmodel)
actModel = ActuationModelFreeFloating(rmodel)
contactModel = ContactModelMultiple(rmodel)
contact6 = ContactModel6D(rmodel,rmodel.getFrameId(contactName),ref=None)
contactModel.addContact(name='contact',contact=contact6)
costModel = CostModelSum(rmodel,nu=actModel.nu)
cost1 = CostModelPosition(rmodel,nu=actModel.nu,
                          frame=rmodel.getFrameId(opPointName),
                          ref=np.array([.5,.4,.3]),
                          activation=ActivationModelWeightedQuad(np.array([1,0,1])))
cost2 = CostModelState(rmodel,State,ref=State.zero(),nu=actModel.nu)
cost3 = CostModelControl(rmodel,nu=actModel.nu)
costModel.addCost( name="pos", weight = 10, cost = cost1)
costModel.addCost( name="regx", weight = 0.1, cost = cost2) 
costModel.addCost( name="regu", weight = 0.01, cost = cost3)

dmodel = DifferentialActionModelFloatingInContact(rmodel,actModel,contactModel,costModel)
model  = IntegratedActionModelEuler(dmodel)
data = model.createData()

cd1 = data.differential.costs .costs['pos']
cd2 = data.differential.costs .costs['regx']
cd3 = data.differential.costs .costs['regu']

ddata = data.differential
rdata = data.differential.pinocchio


q = robot.q0.copy()
v = zero(rmodel.nv)
x = m2a(np.concatenate([q,v]))


# --- DDP 
# --- DDP 
# --- DDP 
from refact import ShootingProblem, SolverDDP,SolverKKT
from logger import *
disp = lambda xs: disptraj(robot,xs)

DT = 1.0 
T = 50
model.timeStep = DT/T

model.differential.costs['pos' ].weight = 10
model.differential.costs['regx'].weight = 0.01
model.differential.costs['regu'].weight = 1
cost1.ref[:] = [ .1, .1, 0.0 ]

import copy
termmodel = copy.copy(model)

termmodel.differential.costs.addCost(name='veleff',cost=CostModelPlacementVelocity(rmodel,cost1.frame),weight=10000)

termmodel.differential.costs['veleff' ].weight = 1000
termmodel.differential.costs['pos' ]   .weight = 300000
termmodel.differential.costs['regx']   .weight = 1
termmodel.differential.costs['regu']   .weight = 0.01
termmodel.differential.costs['regx'].cost.weights = np.array([0]*6+[0.01]*(rmodel.nv-6)+[10]*rmodel.nv)
termmodel.differential.costs['pos'].cost.ref[:] = [ .2, .2, 0 ]
    
# --- SOLVER

problem = ShootingProblem(x, [ model ]*T, termmodel)

ddp = SolverDDP(problem)
ddp.callback = SolverLogger(robot)
ddp.th_stop = 1e-9
ddp.solve(verbose=True,maxiter=1000,regInit=.1)


# --- Contact velocity
# cost = || v ||

