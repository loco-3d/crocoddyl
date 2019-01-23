import pinocchio
from pinocchio.utils import *
from numpy.linalg import inv,pinv,norm,svd,eig
from numpy import dot,asarray
import warnings
from crocoddyl import CostModelSum,CostModelPosition,CostModelState,CostModelControl,DifferentialActionModelFloatingInContact,IntegratedActionModelEuler,ActuationModelFreeFloating,StatePinocchio,ContactModel6D,ContactModelMultiple,ActivationModelWeightedQuad,m2a,a2m,CostModelPlacementVelocity,CostModelPosition6D,ImpulseModelMultiple
from robots import loadTalosLegs

robot = loadTalosLegs()
rmodel = robot.model

rightFrame = 'right_sole_link'
leftFrame = 'left_sole_link'

RIGHTFRAME = rmodel.getFrameId(rightFrame)
LEFTFRAME  = rmodel.getFrameId(leftFrame)

FOOTGAP = .085

def createModel(timeStep,footRef,contactName,opPointName):
    State = StatePinocchio(rmodel)
    actModel = ActuationModelFreeFloating(rmodel)
    contactModel = ContactModelMultiple(rmodel)
    contact6 = ContactModel6D(rmodel,rmodel.getFrameId(contactName),ref=None)
    contactModel.addContact(name='contact',contact=contact6)
    costModel = CostModelSum(rmodel,nu=actModel.nu)
    cost1 = CostModelPosition6D(rmodel,nu=actModel.nu,
                                frame=rmodel.getFrameId(opPointName),
                                ref=pinocchio.SE3(eye(3),np.matrix(footRef).T))
    cost2 = CostModelState(rmodel,State,ref=State.zero(),nu=actModel.nu)
    cost2.weights = np.array([0]*6+[0.01]*(rmodel.nv-6)+[10]*rmodel.nv)
    cost3 = CostModelControl(rmodel,nu=actModel.nu)
    costModel.addCost( name="pos", weight = 100, cost = cost1)
    costModel.addCost( name="regx", weight = 0.1, cost = cost2) 
    costModel.addCost( name="regu", weight = 0.001, cost = cost3)
    
    dmodel = DifferentialActionModelFloatingInContact(rmodel,actModel,contactModel,costModel)
    model  = IntegratedActionModelEuler(dmodel)
    model.timeStep = timeStep
    return model

def createTermModel(timeStep,footRef,contactName,opPointName):
    termmodel = createModel(timeStep,footRef,contactName,opPointName)
    termmodel.differential.costs.addCost(name='veleff',
                                         cost=CostModelPlacementVelocity(rmodel,
                                                                         rmodel.getFrameId(opPointName)),
                                         weight=10000)
    termmodel.differential.costs['veleff' ].weight = 100000
    termmodel.differential.costs['pos' ]   .weight = 100000
    termmodel.differential.costs['regx']   .weight = 1
    termmodel.differential.costs['regu']   .weight = 0.01
    return termmodel

q = robot.q0.copy()
v = zero(rmodel.nv)
x = m2a(np.concatenate([q,v]))


# --- DDP 
# --- DDP 
# --- DDP 
from refact import ShootingProblem, SolverDDP,SolverKKT
from logger import *
disp = lambda xs: disptraj(robot,xs)

DT = 1.
T = 20
timeStep = float(DT)/T

models1 = [ createModel(timeStep=timeStep,
                        footRef = [ (.2*k)/T, FOOTGAP, 0.0 ],
                        contactName = rightFrame,
                        opPointName = leftFrame) for k in range(T) ]
termmodel1 = createTermModel(timeStep=timeStep,footRef = [ .2, FOOTGAP, 0.0 ],
                             contactName = rightFrame,opPointName = leftFrame)
models2 = [ createModel(timeStep=timeStep,
                        footRef = [ (.2*k)/T, -FOOTGAP, 0.0 ],
                        contactName = leftFrame,
                        opPointName = rightFrame) for k in range(T) ]
termmodel2 = createTermModel(timeStep=timeStep,footRef = [ .2, -FOOTGAP, 0.0 ],
                             contactName = leftFrame,opPointName = rightFrame)

# --- SOLVER

'''
termmodel1.timeStep=0
problem1 = ShootingProblem(x, models1, termmodel1 )
ddp1 = SolverDDP(problem1)
#ddp1.callback = SolverLogger(robot)
ddp1.th_stop = 1e-9
ddp1.solve(verbose=True,maxiter=3,regInit=.1)


problemi = ShootingProblem(x, models1, impact1 )
ddpi = SolverDDP(problemi)
#ddpi.callback = SolverLogger(robot)
ddpi.th_stop = 1e-9
ddpi.solve(verbose=True,maxiter=50,regInit=.1,
           init_xs=ddp1.xs,
           init_us=ddp1.us)

problem2 = ShootingProblem(ddp1.xs[-1], models2, termmodel2 )
ddp2 = SolverDDP(problem2)
ddp2.callback = SolverLogger(robot)
ddp2.th_stop = 1e-9
ddp2.solve(verbose=True,maxiter=1000,regInit=.1)
'''

# --- WITH DT=0 at the terminal of the phase
termmodel1.timeStep=0
termmodel2.timeStep=0

models = models1 + [ termmodel1 ] + models2
termmodel = termmodel2

problem = ShootingProblem(x, models, termmodel )
ddp = SolverDDP(problem)
ddp.callback = SolverLogger(robot)
ddp.th_stop = 1e-9
ddp.solve(verbose=True,maxiter=1000,regInit=.1)

assert( norm(ddp.datas()[T].differential.costs['pos'].residuals) < 1e-2 )
assert( norm(ddp.datas()[T].differential.costs['veleff'].residuals) < 5e-3 )
assert( norm(ddp.datas()[-1].differential.costs['pos'].residuals) < 1e-2 )
assert( norm(ddp.datas()[-1].differential.costs['veleff'].residuals) < 5e-3 )


