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
rmodel = robot.model

opPointName = 'right_sole_link'
contactName = 'left_sole_link'

opPointName,contactName = contactName,opPointName
CONTACTFRAME = rmodel.getFrameId(contactName)
OPPOINTFRAME = rmodel.getFrameId(opPointName)

def createModel():
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
    return model

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
T = 20
timeStep = DT/T

models = [ createModel() for _ in range(T+1) ]

for k,model in enumerate(models[:-1]):
    t = k*timeStep
    model.timeStep = timeStep
    model.differential.costs['pos' ].weight = 1
    model.differential.costs['regx'].weight = .1
    model.differential.costs['regx'].cost.weights = np.array([0]*6+[0.01]*(rmodel.nv-6)+[10]*rmodel.nv)
    model.differential.costs['regu'].weight = 0.001
    model.differential.costs['pos' ].cost.ref[:] = [ .2*t/DT, .2*t/DT, 0.0 ]

termmodel = models[-1]
termmodel.differential.costs.addCost(name='veleff',
                                     cost=CostModelPlacementVelocity(rmodel,OPPOINTFRAME),
                                     weight=10000)

termmodel.differential.costs['veleff' ].weight = 100
termmodel.differential.costs['pos' ]   .weight = 3000000
termmodel.differential.costs['regx']   .weight = 1
termmodel.differential.costs['regu']   .weight = 0.01
termmodel.differential.costs['regx'].cost.weights = np.array([0]*6+[0.01]*(rmodel.nv-6)+[10]*rmodel.nv)
termmodel.differential.costs['pos'].cost.ref[:] = [ .2, .2, 0 ]
    
# --- SOLVER

problem = ShootingProblem(x, models[:-1], models[-1] )

ddp = SolverDDP(problem)
ddp.callback = SolverLogger(robot)
ddp.th_stop = 1e-19
ddp.solve(verbose=True,maxiter=1000,regInit=.1)


# --- Contact velocity
# cost = || v ||

