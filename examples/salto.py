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
         +  [ runningModel([ ],{},integrationStep=5e-2) for i in range(6)] \
         +  [ runningModel([ ],{}, com=com0+[0,0,0.9],integrationStep=5e-2) ] \
         +  [ runningModel([ ],{},integrationStep=5e-2) for i in range(8) ] \
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

ddp.solve(maxiter=4000,regInit=.1, #! 400
          init_xs=[rmodel.defaultState]*len(ddp.models()),
          init_us=us0[:imp])
#disp(ddp.xs)

#np.save('jump0.xs',ddp.xs);np.save('jump0.us',ddp.us)
#ddp.xs = np.load('jump0.xs.npy'); ddp.us = np.load('jump0.us.npy')

# ---
'''
from pinocchio.utils import se3ToXYZQUAT
D = 2*np.pi/(imp-9)
for i,x in enumerate(ddp.xs[9:imp]):
    """
    oM1 1_p
    p st [ R p ] 0^c = c = R c + p   => p = c-Rc
    """
    q = a2m(x)[:rmodel.nq]
    pinocchio.centerOfMass(rmodel,rdata,q)
    R = rotate('y',D*i)
    c = rdata.com[0].copy()
    p = c-R*c
    M = SE3(R,p)
    x[:7] = se3ToXYZQUAT(M*rdata.oMi[1])
    

models[high] = runningModel([],
                             { rightId: SE3(eye(3), right0),
                               leftId: SE3(eye(3), left0) },integrationStep=3e-2)

models[high].differential.costs['xreg'].cost.ref = ddp.xs[high].copy()
models[high].differential.costs['xreg'].cost.activation.weights[:6] = 100

models[high].differential.costs['track16'].cost.activation = ActivationModelInequality(np.array([-.05,-.05,0,-np.inf,-np.inf,-np.inf]),np.array([ .05, .05,np.inf,np.inf,np.inf,np.inf]))
models[high].differential.costs['track16'].cost.ref.translation += np.matrix([0.,0,1]).T
models[high].differential.costs['track30'].cost.activation = ActivationModelInequality(np.array([-.05,-.05,0,-np.inf,-np.inf,-np.inf]),np.array([ .05, .05,np.inf,np.inf,np.inf,np.inf]))
models[high].differential.costs['track30'].cost.ref.translation += np.matrix([0,0,1]).T


problem.runningModels[high] = models[high]
problem.runningDatas[high] = models[high].createData()

trajs = {}
for i in range(6):
    models[high].differential.costs['xreg'].weight=10**i
    models[high].differential.costs['track16'].weight=10**i
    models[high].differential.costs['track30'].weight=10**i
    ddp.solve(maxiter=2000,regInit=.1,
              init_xs=ddp.xs,
              init_us=ddp.us, isFeasible=False)
    trajs[i] = [ x.copy() for x in ddp.xs ]
'''



from pinocchio.utils import se3ToXYZQUAT
D = 2*np.pi/(imp-9)
refs = []
for i,x in enumerate(ddp.xs[9:imp]):
    '''
    oM1 1_p
    p st [ R p ] 0^c = c = R c + p   => p = c-Rc
    '''
    q = a2m(x)[:rmodel.nq]
    pinocchio.centerOfMass(rmodel,rdata,q)
    R = rotate('y',D*i)
    c = rdata.com[0].copy()
    p = c-R*c
    M = SE3(R,p)
    refs.append( x.copy() )
    refs[-1][:7] = se3ToXYZQUAT(M*rdata.oMi[1])


#for i,m in enumerate(models[9:imp]):
#    m.differential.costs['xreg'].weight = 10
#    m.differential.costs['xreg'].cost.ref = refs[i]
#    m.differential.costs['xreg'].cost.activation.weights[3:6] = 10
models[imp].costs['xreg'].cost.activation.weights[3:6] = 100

ANG = .5
cos,sin = np.cos,np.sin
models[imp].costs['xreg'].cost.ref[3:7] = [ 0,sin(ANG),0,cos(ANG) ]

ddp.callback.append(CallbackSolverDisplay(robot,rate=5,freq=10))
for i in range(1,7):
    #for k in range(9,imp):
    impact.costs['xreg'].cost.activation.weights[3:rmodel.nv] = 10**i
    ddp.solve(maxiter=2000,regInit=.1, #!2000
              init_xs=ddp.xs,
              init_us=ddp.us, isFeasible=False)

# for i,m in enumerate(models[9:imp]):
#     m.differential.costs['xreg'].cost.activation.weights[7] = 10
#     m.differential.costs['xreg'].cost.activation.weights[13] = 10
# models[high].differential.costs['xreg'].cost.activation.weights[7] = 100
# models[high].differential.costs['xreg'].cost.activation.weights[13] = 100
    
jumps = []
for ANG in np.arange(.5,3.2,.3):
    models[imp].costs['xreg'].cost.ref[3:7] = [ 0,sin(ANG),0,cos(ANG) ]
    ddp.solve(maxiter=2000,regInit=.1, #!2000
              init_xs=ddp.xs,
              init_us=ddp.us, isFeasible=True)
    #!disp(ddp.xs)
    jumps.append({ 'x': [x.copy() for x in ddp.xs ],'u': [u.copy() for u in ddp.us ] })
    
#ddp.xs = np.load('salto.xs.npy')
#ddp.us = np.load('salto.us.npy')

ddp.th_stop = 5e-4
impact.costs['track30'].weight = 1e6
impact.costs['track16'].weight = 1e6
# for i in range(9,imp):
#      models[i].costs['xreg'].cost.activation.weights[7:rmodel.nv] = .5
 
#
ddp.solve(init_xs=ddp.xs,init_us=ddp.us,maxiter=1,isFeasible=True)
disp(ddp.xs)

#models[high].differential.costs['xreg'].cost.activation.weights[6:rmodel.nv] = 100000 ## obtain an fliped-axel


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


