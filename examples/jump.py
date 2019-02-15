'''
Example of Crocoddyl sequence for computing a whole-body jump. 
The sequence is composed of a double-support phase, a flight phase, a double impact and a 
double-support landing phase. The main cost are an elevation of the COM at the middle of the flight
phase, a minimization of the COM energy loss at impact, and a terminal stabilization cost
in the configuration space. Additional terms are added for regularization.

The tricky part comes from the search which is done in two parts: in a first part, we look for 
a feasible solution for jumping, without considering landing constraints. Once a jump is
accepted, we put the second part of the sequence (landing) and add constraints on the landing.

Baseline: the first part of the search takes 20 iteration, the second part takes 40 iterations, the
com elevation is 20cm. It seems difficult to increase the time of the jump because it makes
the trajectory optimization too unstable.
'''

from crocoddyl import StatePinocchio
from crocoddyl import DifferentialActionModelFloatingInContact
from crocoddyl import IntegratedActionModelEuler
from crocoddyl import CostModelSum
from crocoddyl import CostModelFramePlacement, CostModelFrameVelocity
from crocoddyl import CostModelState, CostModelControl,CostModelCoM
from crocoddyl import ActivationModelWeightedQuad,ActivationModelInequality
from crocoddyl import ActuationModelFreeFloating
from crocoddyl import ContactModel6D, ContactModelMultiple
from crocoddyl import ShootingProblem
from crocoddyl import SolverDDP
from crocoddyl import CallbackDDPLogger, CallbackDDPVerbose, CallbackSolverDisplay
from crocoddyl import plotOCSolution, plotDDPConvergence
from crocoddyl import loadTalosLegs
from crocoddyl import m2a, a2m, ImpulseModelMultiple, ImpulseModel6D, ActionModelImpact
from crocoddyl.impact import CostModelImpactCoM, CostModelImpactWholeBody

import pinocchio
from pinocchio.utils import *
from numpy.linalg import norm,inv,pinv,eig,svd

robot = loadTalosLegs()
robot.model.armature[6:] = .3
robot.initDisplay(loadModel=True)

rmodel = robot.model
rdata  = rmodel.createData()

# Setting up the 3d walking problem
rightFoot = 'right_sole_link'
leftFoot = 'left_sole_link'

rightId = rmodel.getFrameId(rightFoot)
leftId  = rmodel.getFrameId(leftFoot)

# Create the initial state
q0 = robot.q0.copy()
v0 = zero(rmodel.nv)
x0 = m2a(np.concatenate([q0,v0]))
rmodel.defaultState = x0.copy()

# Solving the 3d walking problem using DDP
stepLength = 0.2
swingDuration = 0.75
stanceDurantion = 0.1

from crocoddyl.diagnostic import displayTrajectory
disp = lambda xs,dt: displayTrajectory(robot,xs,dt)
disp.__defaults__ = ( .1, )

def runningModel(contactIds, effectors, com=None, integrationStep = 1e-2):
    '''
    Creating the action model for floating-base systems. A walker system 
    is by default a floating-base system.
    contactIds is a list of frame Ids of points that should be in contact.
    effectors is a dict of key frame ids and SE3 values of effector references.
    '''
    actModel = ActuationModelFreeFloating(rmodel)
    State = StatePinocchio(rmodel)

    # Creating a 6D multi-contact model, and then including the supporting foot
    contactModel = ContactModelMultiple(rmodel)
    for cid in contactIds:
       contactModel.addContact('contact%d'%cid,
                               ContactModel6D(rmodel, cid, ref=None))

    # Creating the cost model for a contact phase
    costModel = CostModelSum(rmodel, actModel.nu)
    wx = np.array([0]*6 + [.1]*(rmodel.nv-6) + [10]*rmodel.nv)
    costModel.addCost('xreg',weight=1e-1,
                      cost=CostModelState(rmodel,State,ref=rmodel.defaultState,nu=actModel.nu,
                                          activation=ActivationModelWeightedQuad(wx)))
    costModel.addCost('ureg',weight=1e-4,
                      cost=CostModelControl(rmodel, nu=actModel.nu))
    for fid,ref in effectors.items():
        costModel.addCost("track%d"%fid, weight=100.,
                          cost = CostModelFramePlacement(rmodel,fid,ref,actModel.nu))

    if com is not None:
        costModel.addCost("com", weight=10000.,
                          cost = CostModelCoM(rmodel,ref=com,nu=actModel.nu))
        
    # Creating the action model for the KKT dynamics with simpletic Euler
    # integration scheme
    dmodel = \
             DifferentialActionModelFloatingInContact(rmodel,
                                                      actModel,
                                                      contactModel,
                                                      costModel)
    model = IntegratedActionModelEuler(dmodel)
    model.timeStep = integrationStep
    return model

def pseudoImpactModel(contactIds,effectors):
    #assert(len(effectors)==1)
    model = runningModel(contactIds,effectors,integrationStep=0)

    costModel = model.differential.costs
    for fid,ref in effectors.items():
        costModel.addCost('impactVel%d' % fid,weight = 100.,
                          cost=CostModelFrameVelocity(rmodel,fid))
        costModel.costs['track%d'%fid ].weight = 100
    costModel.costs['xreg'].weight = 1
    costModel.costs['ureg'].weight = 0.01

    return model

def impactModel(contactIds,effectors):
    State = StatePinocchio(rmodel)

    # Creating a 6D multi-contact model, and then including the supporting foot
    impulseModel = ImpulseModelMultiple(rmodel,{ "impulse%d"%cid: ImpulseModel6D(rmodel,cid)
                                                 for cid in contactIds })

    # Creating the cost model for a contact phase
    costModel = CostModelSum(rmodel,nu=0)
    wx = np.array([0]*6 + [.1]*(rmodel.nv-6) + [10]*rmodel.nv)
    costModel.addCost('xreg',weight=.1,
                      cost=CostModelState(rmodel,State,ref=rmodel.defaultState,nu=0,
                                          activation=ActivationModelWeightedQuad(wx)))
    costModel.addCost('com',weight=1.,
                      cost=CostModelImpactCoM(rmodel,
                                              activation=ActivationModelWeightedQuad(m2a([.1,.1,3.]))))
    for fid,ref in effectors.items():
        costModel.addCost("track%d"%fid, weight=100.,
                          cost = CostModelFramePlacement(rmodel,fid,ref,nu=0))
        # costModel.addCost("vel%d"%fid, weight=0.,
        #                   cost = CostModelFrameVelocity(rmodel,fid,nu=0))
        
    # Creating the action model for the KKT dynamics with simpletic Euler
    # integration scheme
    model = \
             ActionModelImpact(rmodel,impulseModel,costModel)
    return model


### --- MODEL SEQUENCE 
### --- MODEL SEQUENCE 
### --- MODEL SEQUENCE 
SE3 = pinocchio.SE3
pinocchio.forwardKinematics(rmodel,rdata,q0)
pinocchio.updateFramePlacements(rmodel,rdata)
right0 = rdata.oMf[rightId].translation
left0  = rdata.oMf[leftId].translation
com0 = m2a(pinocchio.centerOfMass(rmodel,rdata,q0))

models =\
         [ runningModel([ rightId, leftId ],{}, integrationStep=4e-2) for i in range(10) ] \
         +  [ runningModel([ ],{},integrationStep=3e-2) for i in range(5)] \
         +  [ runningModel([ ],{}, com=com0+[0,0,0.5],integrationStep=5e-2) ] \
         +  [ runningModel([ ],{},integrationStep=3e-2) for i in range(7) ] \
         +  [ impactModel([ leftId,rightId ], 
                          { rightId: SE3(eye(3), right0),
                            leftId: SE3(eye(3), left0) }) ] \
        +  [ runningModel([ rightId, leftId ],{},integrationStep=2e-2) for i in range(9)] \
        +  [ runningModel([ rightId, leftId ],{}, integrationStep=0) ]

high = [ isinstance(m,IntegratedActionModelEuler) and 'com' in m.differential.costs.costs for m in models ].index(True)
models[high].differential.costs['com'].cost.activation = ActivationModelInequality(np.array([-.01,-.01,-0.01]),np.array([ .01, .01,0.1]))

imp = [ isinstance(m,ActionModelImpact) for m in models ].index(True)
impact = models[imp]
impact.costs['track30'].weight=0
impact.costs['track16'].weight=0
impact.costs['com'].weight=100
impact.costs['track16'].cost.activation = ActivationModelWeightedQuad(np.array([.2,1,.1,1,1,1]))
impact.costs['track30'].cost.activation = ActivationModelWeightedQuad(np.array([.2,1,.1,1,1,1]))
impact.costs.addCost(name='xycom',cost=CostModelCoM(rmodel,ref=com0,activation=ActivationModelWeightedQuad(np.array([1.,.2,0]))),weight=10)

for m in models[imp+1:]:
    m.differential.costs['xreg'].weight = 0.0
    m.differential.contact['contact16'].gains[1] = 30
    m.differential.contact['contact30'].gains[1] = 30

models[-1].differential.costs['xreg'].weight = 1000
models[-1].differential.costs['xreg'].cost.activation.weights[:] = 1


# ---------------------------------------------------------------------------------------------
# Solve both take-off and landing.
# Solve the initial phase (take-off).
problem = ShootingProblem(initialState=x0,runningModels=models[:imp],terminalModel=models[imp])
ddp = SolverDDP(problem)
ddp.callback = [ CallbackDDPLogger(), CallbackDDPVerbose() ]#, CallbackSolverDisplay(robot,rate=5) ]
ddp.th_stop = 1e-4
us0 = [ m.differential.quasiStatic(d.differential,rmodel.defaultState) \
        for m,d in zip(ddp.models(),ddp.datas())[:imp] ] \
            +[np.zeros(0)]+[ m.differential.quasiStatic(d.differential,rmodel.defaultState) \
                             for m,d in zip(ddp.models(),ddp.datas())[imp+1:-1] ]

ddp.solve(maxiter=20,regInit=.1,
          init_xs=[rmodel.defaultState]*len(ddp.models()),
          init_us=us0[:imp])
disp(ddp.xs)

# ---------------------------------------------------------------------------------------------
# Solve both take-off and landing.
xsddp = ddp.xs
usddp = ddp.us

problem = ShootingProblem(initialState=x0,runningModels=models[:-1],terminalModel=models[-1])
ddp = SolverDDP(problem)
ddp.callback = [ CallbackDDPLogger(), CallbackDDPVerbose() ]#, CallbackSolverDisplay(robot,rate=5,freq=10) ]

ddp.xs = xsddp + [rmodel.defaultState]*(len(models)-len(xsddp))
ddp.us = usddp + [ np.zeros(0) if isinstance(m,ActionModelImpact) else \
               m.differential.quasiStatic(d.differential,rmodel.defaultState) \
                               for m,d in zip(ddp.models(),ddp.datas())[len(usddp):-1] ]

ddp.th_stop = 1e-1
ddp.solve(init_xs=ddp.xs,init_us=ddp.us)


ddp.th_stop = 5e-4
impact.costs['track30'].weight = 1e6
impact.costs['track16'].weight = 1e6
ddp.solve(init_xs=ddp.xs,init_us=ddp.us,maxiter=100,isFeasible=True)
disp(ddp.xs)


xjump0 = [ x.copy() for x in ddp.xs ]
ujump0 = [ u.copy() for u in ddp.us ]

'''
### Jump with frontal scissors.
fig = high+2
x = x0.copy()
x[9 ] =  1.
x[15] = -1.
models[fig].differential.costs.costs['xreg'].cost.ref=x.copy()
models[fig].differential.costs.costs['xreg'].cost.activation.weights[rmodel.nv:] = 0

for i in range(6,9):
    impact.costs['track30'].weight = 10**i
    impact.costs['track16'].weight = 10**i
    models[fig].differential.costs.costs['xreg'].weight  = 10**i
    ddp.solve(init_xs=ddp.xs,init_us=ddp.us,maxiter=100,isFeasible=True)
    disp(ddp.xs)

xjump1 = [ x.copy() for x in ddp.xs ]
ujump1 = [ u.copy() for u in ddp.us ]
ddp.xs = xjump0
ddp.us = ujump0

### Jump with lateral scissors.
fig = high+2
x = x0.copy()
x[8 ] =  .8
x[14] = -.8
models[fig].differential.costs.costs['xreg'].cost.ref=x.copy()
models[fig].differential.costs.costs['xreg'].cost.activation.weights[rmodel.nv:] = 0

for i in range(6,9):
    impact.costs['track30'].weight = 10**i
    impact.costs['track16'].weight = 10**i
    models[fig].differential.costs.costs['xreg'].weight  = 10**i
    ddp.solve(init_xs=ddp.xs,init_us=ddp.us,maxiter=100,isFeasible=True)
    disp(ddp.xs)
    
xjump2 = [ x.copy() for x in ddp.xs ]
ujump2 = [ u.copy() for u in ddp.us ]
ddp.xs = xjump0
ddp.us = ujump0

models[fig].differential.costs.costs['xreg'].weight  = 100
models[fig].differential.costs.costs['xreg'].cost.ref=x0.copy()
models[fig].differential.costs.costs['xreg'].cost.activation.weights[rmodel.nv:] = 10
'''

'''
### Jump with twist PI/2
impact.costs['track16'].cost.ref = SE3(rotate('z',1.5),zero(3))*impact.costs['track16'].cost.ref
impact.costs['track30'].cost.ref = SE3(rotate('z',1.5),zero(3))*impact.costs['track30'].cost.ref
models[-1].differential.costs.costs['xreg'].cost.activation.weights[5] = 0

for i in range(4,9):
    impact.costs['track30'].weight = 10**i
    impact.costs['track16'].weight = 10**i
    ddp.solve(init_xs=ddp.xs,init_us=ddp.us,maxiter=100,isFeasible=True)
    disp(ddp.xs)
    
xjump3 = [ x.copy() for x in ddp.xs ]
ujump3 = [ u.copy() for u in ddp.us ]
#ddp.xs = xjump0
#ddp.us = ujump0

### Jump with twist PI
impact.costs['track16'].cost.ref = SE3(rotate('z',3.15),zero(3))*impact.costs['track16'].cost.ref
impact.costs['track30'].cost.ref = SE3(rotate('z',3.15),zero(3))*impact.costs['track30'].cost.ref
models[-1].differential.costs.costs['xreg'].cost.activation.weights[5] = 0

for i in range(4,9):
    impact.costs['track30'].weight = 10**i
    impact.costs['track16'].weight = 10**i
    ddp.solve(init_xs=ddp.xs,init_us=ddp.us,maxiter=100,isFeasible=True)
    disp(ddp.xs)
    
xjump4 = [ x.copy() for x in ddp.xs ]
ujump4 = [ u.copy() for u in ddp.us ]
ddp.xs = xjump0
ddp.us = ujump0

'''

### Salto!
impact.costs['track16'].cost.ref = SE3(eye(3),left0)
impact.costs['track30'].cost.ref = SE3(eye(3),right0)
models[-1].differential.costs.costs['xreg'].cost.activation.weights[5] = 1

xsalto = models[high].differential.costs.costs['xreg'].cost.ref.copy()
xsalto[3]=1
xsalto[6]=0
models[high].differential.costs.costs['xreg'].cost.ref = xsalto
models[high].differential.costs.costs['xreg'].cost.activation.weights[3:5] = 100

impact.costs['track30'].weight = 0.1
impact.costs['track16'].weight = 0.1
ddp.xs = [x.copy() for x in xjump0]
ddp.us = [u.copy() for u in ujump0]

# from pinocchio.utils import se3ToXYZQUAT
# D = 2*np.pi/(imp-9)
# for i,x in enumerate(ddp.xs[9:imp]):
#     '''
#     oM1 1_p
#     p st [ R p ] 0^c = c = R c + p   => p = c-Rc
#     '''
#     q = a2m(x)[:rmodel.nq]
#     pinocchio.centerOfMass(rmodel,rdata,q)
#     R = rotate('y',D*i)
#     c = rdata.com[0].copy()
#     p = c-R*c
#     M = SE3(R,p)
#     x[:7] = se3ToXYZQUAT(M*rdata.oMi[1])
    
    


for i in range(1,9):
    # impact.costs['track30'].weight = 10**i
    # impact.costs['track16'].weight = 10**i
    models[high].differential.costs.costs['xreg'].weight = 10**i
    ddp.solve(init_xs=ddp.xs,init_us=ddp.us,maxiter=100,isFeasible=False)
    disp(ddp.xs)
    
xjump5 = [ x.copy() for x in ddp.xs ]
ujump5 = [ u.copy() for u in ddp.us ]
ddp.xs = xjump0
ddp.us = ujump0
