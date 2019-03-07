import warnings

from numpy import asarray, dot, matrix
from numpy.linalg import eig, inv, norm, pinv, svd

import pinocchio
from crocoddyl import (ActivationModelWeightedQuad, ActuationModelFreeFloating, ContactModel6D, ContactModelMultiple,
                       CostModelControl, CostModelPlacementVelocity, CostModelPosition, CostModelPosition6D,
                       CostModelState, CostModelSum, DifferentialActionModelFloatingInContact, ImpulseModelMultiple,
                       IntegratedActionModelEuler, StatePinocchio, a2m, m2a)
from crocoddyl.impact import ActionModelImpact, ImpulseModel6D
from logger import *
from pinocchio.utils import *
# --- DDP
# --- DDP
# --- DDP
from refact import ShootingProblem, SolverDDP, SolverKKT
from robots import loadTalosLegs

robot = loadTalosLegs()
rmodel = robot.model

rightFrame = 'right_sole_link'
leftFrame = 'left_sole_link'

RIGHTFRAME = rmodel.getFrameId(rightFrame)
LEFTFRAME = rmodel.getFrameId(leftFrame)

FOOTGAP = .085


def createModel(timeStep, footRef, contactName, opPointName):
    State = StatePinocchio(rmodel)
    actModel = ActuationModelFreeFloating(rmodel)
    contactModel = ContactModelMultiple(rmodel)
    contact6 = ContactModel6D(rmodel, rmodel.getFrameId(contactName), ref=pinocchio.SE3.Identity(), gains=[0., 0.])
    contactModel.addContact(name='contact', contact=contact6)
    costModel = CostModelSum(rmodel, nu=actModel.nu)
    cost1 = CostModelPosition6D(
        rmodel,
        nu=actModel.nu,
        frame=rmodel.getFrameId(opPointName),
        ref=pinocchio.SE3(eye(3),
                          np.matrix(footRef).T),
        gains=[0., 0.])
    cost2 = CostModelState(rmodel, State, ref=State.zero(), nu=actModel.nu)
    cost2.weights = np.array([0] * 6 + [0.01] * (rmodel.nv - 6) + [10] * rmodel.nv)
    cost3 = CostModelControl(rmodel, nu=actModel.nu)
    costModel.addCost(name="pos", weight=100, cost=cost1)
    costModel.addCost(name="regx", weight=0.1, cost=cost2)
    costModel.addCost(name="regu", weight=0.001, cost=cost3)

    dmodel = DifferentialActionModelFloatingInContact(rmodel, actModel, contactModel, costModel)
    model = IntegratedActionModelEuler(dmodel)
    model.timeStep = timeStep
    return model


def createTermModel(timeStep, footRef, contactName, opPointName):
    termmodel = createModel(timeStep, footRef, contactName, opPointName)
    termmodel.differential.costs.addCost(
        name='veleff', cost=CostModelPlacementVelocity(rmodel, rmodel.getFrameId(opPointName)), weight=10000)
    termmodel.differential.costs['veleff'].weight = 100000
    termmodel.differential.costs['pos'].weight = 100000
    termmodel.differential.costs['regx'].weight = 1
    termmodel.differential.costs['regu'].weight = 0.01
    return termmodel


q = robot.q0.copy()
v = zero(rmodel.nv)
x = m2a(np.concatenate([q, v]))

disp = lambda xs: disptraj(robot, xs)

DT = 1.
T = 20
timeStep = float(DT) / T

models1 = [
    createModel(
        timeStep=timeStep, footRef=[(.2 * k) / T, FOOTGAP, 0.0], contactName=rightFrame, opPointName=leftFrame)
    for k in range(T)
]
termmodel1 = createTermModel(
    timeStep=timeStep, footRef=[.2, FOOTGAP, 0.0], contactName=rightFrame, opPointName=leftFrame)
models2 = [
    createModel(
        timeStep=timeStep, footRef=[(.2 * k) / T, -FOOTGAP, 0.0], contactName=leftFrame, opPointName=rightFrame)
    for k in range(T)
]
termmodel2 = createTermModel(
    timeStep=timeStep, footRef=[.2, -FOOTGAP, 0.0], contactName=leftFrame, opPointName=rightFrame)

termcostModel = CostModelSum(rmodel, nu=0)
termcost1 = CostModelPosition6D(
    rmodel, nu=0, frame=LEFTFRAME, ref=pinocchio.SE3(eye(3),
                                                     np.matrix([.2, FOOTGAP, 0]).T))
termcost2 = CostModelState(rmodel, StatePinocchio(rmodel), ref=StatePinocchio(rmodel).zero(), nu=0)
termcost2.weights = np.array([0] * 6 + [0.01] * (rmodel.nv - 6) + [10] * rmodel.nv)
termcostModel.addCost(name="pos", weight=1000000, cost=termcost1)
termcostModel.addCost(name="regx", weight=1, cost=termcost2)

impulseModelL = ImpulseModel6D(rmodel, LEFTFRAME)
impulseModelR = ImpulseModel6D(rmodel, RIGHTFRAME)
impulseModel = ImpulseModelMultiple(rmodel, {'right': impulseModelR, 'left': impulseModelL})

impact1 = ActionModelImpact(rmodel, impulseModel, termcostModel)
impact2 = ActionModelImpact(rmodel, impulseModel, termcostModel)

# --- WITH IMPACT MODEL

for m in models1:
    m.differential.costs['pos'].weight = 1
for m in models2:
    m.differential.costs['pos'].weight = 1

problem = ShootingProblem(x, models1, impact1)
ddp = SolverDDP(problem)
#ddp.callback = [CallbackDDPLogger()]
ddp.th_stop = 1e-9
ddp.solve(verbose=True, maxiter=20, regInit=.1)

di = ddp.datas()[-1]
dir = di.impulse['right']
dil = di.impulse['left']
mi = ddp.models()[-1]
mir = di.impulse['right']
mil = di.impulse['left']

model = impact1
nx, ndx, nu, nq, nv, nout, nc = model.nx, model.State.ndx, model.nu, model.nq, model.nv, model.nout, model.nimpulse
q = a2m(x[:nq])
v = a2m(x[-nv:])
vnext = a2m(di.vnext)
M = np.matrix(di.K[:nv, :nv])
J = matrix(di.impulse.J)
f = a2m(di.f)
Jr = J[:6, :]
Jl = J[6:, :]
fr = f[:6, :]
fl = f[6:, :]
