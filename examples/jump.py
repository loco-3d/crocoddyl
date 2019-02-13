from crocoddyl import StatePinocchio
from crocoddyl import DifferentialActionModelFloatingInContact
from crocoddyl import IntegratedActionModelEuler
from crocoddyl import CostModelSum
from crocoddyl import CostModelFramePlacement, CostModelFrameVelocity
from crocoddyl import CostModelState, CostModelControl,CostModelCoM
from crocoddyl import ActivationModelWeightedQuad
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

SE3 = pinocchio.SE3
pinocchio.forwardKinematics(rmodel,rdata,q0)
pinocchio.updateFramePlacements(rmodel,rdata)
right0 = rdata.oMf[rightId].translation
left0  = rdata.oMf[leftId].translation
com0 = m2a(pinocchio.centerOfMass(rmodel,rdata,q0))

# + [ runningModel([ rightId, leftId ],{},com=com0-[0,0,.1], integrationStep=5e-2) ] \
# + [ runningModel([ rightId, leftId ],{},integrationStep=5e-2)] *6\
# + [ runningModel([ rightId, leftId ],{},com=com0-[0,0,0], integrationStep=5e-2) ] \

models =\
         [ runningModel([ rightId, leftId ],{}, integrationStep=5e-2) for i in range(10) ] \
         +  [ runningModel([ ],{},integrationStep=5e-2) for i in range(5)] \
         +  [ runningModel([ ],{}, com=com0+[0,0,0.5],integrationStep=5e-2) ] \
         +  [ runningModel([ ],{},integrationStep=5e-2) for i in range(4) ] \
         +  [ impactModel([ leftId,rightId ], 
                          { rightId: SE3(eye(3), right0),
                            leftId: SE3(eye(3), left0) }) ] \
        +  [ runningModel([ rightId, leftId ],{},integrationStep=2e-2) for i in range(9)] \
        +  [ runningModel([ rightId, leftId ],{}, integrationStep=0) ]

imp = 20
impact = models[imp]
impact.costs['track30'].weight=0
impact.costs['track16'].weight=0
impact.costs['com'].weight=100

for m in models[imp+1:]:
    m.differential.costs['xreg'].weight = 0.0
    m.differential.contact['contact16'].gains[1] = 30
    m.differential.contact['contact30'].gains[1] = 30

models[-1].differential.costs['xreg'].weight = 1000
models[-1].differential.costs['xreg'].cost.activation.weights[:] = 1


problem = ShootingProblem(initialState=x0,runningModels=models[:20],terminalModel=models[20])
ddp = SolverDDP(problem)
ddp.callback = [ CallbackDDPLogger(), CallbackDDPVerbose() ]
ddp.th_stop = 1e-6
us0 = [ m.differential.quasiStatic(d.differential,rmodel.defaultState) \
        for m,d in zip(ddp.models(),ddp.datas())[:imp] ] \
            +[np.zeros(0)]+[ m.differential.quasiStatic(d.differential,rmodel.defaultState) \
                             for m,d in zip(ddp.models(),ddp.datas())[imp+1:-1] ]

ddp.solve(maxiter=1000,regInit=.1,
          init_xs=[rmodel.defaultState]*len(ddp.models()),
          init_us=us0[:20])


# for i in range(21,len(models)):
#     disp(ddp.xs)
#     xs = ddp.xs + [ ddp.xs[-1] ]
#     us = ddp.us + [ np.zeros(problem.terminalModel.nu) ]
#     problem = ShootingProblem(initialState=x0,runningModels=models[:i],terminalModel=models[i])
#     ddp = SolverDDP(problem)
#     ddp.callback = [ CallbackDDPLogger(), CallbackDDPVerbose() ]
#     ddp.th_stop = 1e-6
#     ddp.solve(init_xs=xs,init_us=us)

xsddp = ddp.xs
usddp = ddp.us

problem = ShootingProblem(initialState=x0,runningModels=models[:-1],terminalModel=models[-1])
ddp = SolverDDP(problem)

xs = xsddp + [rmodel.defaultState]*(len(models)-len(xsddp))
us = usddp + [np.zeros(0)] + [ m.differential.quasiStatic(d.differential,rmodel.defaultState) \
                               for m,d in zip(ddp.models(),ddp.datas())[21:-1] ]

ddp.callback = [ CallbackDDPLogger(), CallbackDDPVerbose() ]
ddp.th_stop = 1e-6
ddp.solve(init_xs=xs,init_us=us)




'''
np.set_printoptions(precision=4, linewidth=200, suppress=True)
nq = rmodel.nq
for m,d,x in zip(ddp.models(),ddp.datas(),ddp.xs):
    if isinstance(m,IntegratedActionModelEuler):
        pinocchio.forwardKinematics(rmodel,d.differential.pinocchio,a2m(x[:nq]),a2m(x[nq:]))
        pinocchio.updateFramePlacements(rmodel,d.differential.pinocchio)
        print pinocchio.getFrameVelocity(rmodel,d.differential.pinocchio,rightId).vector.T

import matplotlib.pylab as plt
from crocoddyl.integrated_action import IntegratedActionDataEuler
plt.ion()
#plt.plot([ d.differential.pinocchio.oMf[rightId].translation[2,0] \
#           for d in ddp.datas() if isinstance(d,IntegratedActionDataEuler) ])

plt.plot([ d.differential.pinocchio.com[0][2,0] \
           for d in ddp.datas() if isinstance(d,IntegratedActionDataEuler) ])


for i in range(1,8):
    disp(ddp.xs)
    models[21].differential.costs['ureg'].weight = 0.1*10**i
    models[-1].differential.costs['xreg'].cost.activation.weights[3:6] = 10**i
    models[-1].differential.costs['xreg'].cost.activation.weights[6:] = 10**i
    models[20].costs['track30'].weight=10**i
    models[20].costs['track16'].weight=10**i
    models[20].costs['com'].weight=0.1*10**i
    if i>5:
        impact.impulseWeight = 10**(i-4)
    ddp.solve(maxiter=1000,regInit=.1,init_xs=ddp.xs,init_us=ddp.us)
    for m,d,x in zip(ddp.models(),ddp.datas(),ddp.xs):
      if isinstance(m,IntegratedActionModelEuler):
        pinocchio.forwardKinematics(rmodel,d.differential.pinocchio,a2m(x[:nq]),a2m(x[nq:]))
        pinocchio.updateFramePlacements(rmodel,d.differential.pinocchio)
        print pinocchio.getFrameVelocity(rmodel,d.differential.pinocchio,rightId).vector.T
    #plt.plot([ d.differential.pinocchio.oMf[rightId].translation[2,0] \
    #       for d in ddp.datas() if isinstance(d,IntegratedActionDataEuler) ])

    plt.plot([ d.differential.pinocchio.com[0][2,0] \
           for d in ddp.datas() if isinstance(d,IntegratedActionDataEuler) ])
    

'''
stophere

nq,nv = rmodel.nq,rmodel.nv

models = [] \
        + [ runningModel([ ],{},integrationStep=5e-2) for i in range(4) ] \
        + [ impactModel([ leftId,rightId ], 
                      { rightId: SE3(eye(3), right0),
                        leftId: SE3(eye(3), left0) }) ] \
        +  [ runningModel([ rightId, leftId ],{},integrationStep=2e-2) for i in range(9) ]

imp = 4
impact = models[imp]

for m in models[imp+1:]:
    m.differential.costs['xreg'].weight = 0.0
    m.differential.contact['contact16'].gains[1] = 30
    m.differential.contact['contact30'].gains[1] = 30
    
models[-1].differential.costs['xreg'].weight = 1000
models[-1].differential.costs['xreg'].cost.activation.weights[:] = 1
impact.costs['com'].weight=100
impact.costs['track16'].weight=1000
impact.costs['track30'].weight=1000

x0fall = x0.copy()
x0fall[2] += .1
#x0fall[nq+2] = -3

problem = ShootingProblem(initialState=x0fall,runningModels=models[:-1],terminalModel=models[-1])
ddp = SolverDDP(problem)
ddp.callback = [ CallbackDDPLogger(), CallbackDDPVerbose() ]
ddp.th_stop = 1e-6
ddp.solve(maxiter=1000,regInit=.1,
          init_xs=[rmodel.defaultState]*len(ddp.models()),
          init_us=[ m.differential.quasiStatic(d.differential,rmodel.defaultState) \
                    for m,d in zip(ddp.models(),ddp.datas())[:imp] ] \
                  +[np.zeros(0)]+[ m.differential.quasiStatic(d.differential,rmodel.defaultState) \
                                   for m,d in zip(ddp.models(),ddp.datas())[imp+1:-1] ])

'''
xi[-nv:] = 0
xi[nv+3] = -1
impact.calc(di,xi)

q = a2m(xi[:nq])
vq = a2m(di.vnext)

import time
for i in range(100):
    q = pinocchio.integrate(rmodel,q,vq*1e-2)
    robot.display(q)
    time.sleep(1e-3)
'''

