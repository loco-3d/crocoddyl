'''
Example of Crocoddyl sequence for computing a whole-body (biped) salto.
The training is difficult and hand-tuned. We first work with only the jump phase (first
half of the sequence) and search for a simple jumps, then for a serie of jumps with
increasing terminal angle, to discover the salto. When a 2PI rotation is discovered,
we had the landing phase and converge.

The initial jump is a litteral copy of jump.py
Each increase of the angle takes ~50 iterations to converge. The landing phase takes about ~20
iterations to converge.
'''

import sys

import numpy as np

import pinocchio
from crocoddyl import (ActionModelImpact, ActivationModelInequality, ActivationModelWeightedQuad,
                       ActuationModelFreeFloating, CallbackDDPVerbose, ContactModel6D, ContactModelMultiple,
                       CostModelCoM, CostModelControl, CostModelFramePlacement, CostModelFrameVelocity, CostModelState,
                       CostModelSum, DifferentialActionModelFloatingInContact, ImpulseModel6D, ImpulseModelMultiple,
                       IntegratedActionModelEuler, ShootingProblem, SolverDDP, StatePinocchio, loadTalosLegs, m2a)
from crocoddyl.diagnostic import displayTrajectory
from crocoddyl.impact import CostModelImpactCoM
from pinocchio.utils import eye, zero

# from pinocchio.utils import *

# Number of iterations in each phase. If 0, try to load.
PHASE_ITERATIONS = {"initial": 200, "angle": 200, "landing": 200}
PHASE_BACKUP = {"initial": False, "angle": False, "landing": False}
BACKUP_PATH = "npydata/salto."

if 'load' in sys.argv:
    PHASE_ITERATIONS = {k: 0 for k in PHASE_ITERATIONS}
if 'save' in sys.argv:
    PHASE_BACKUP = {k: True for k in PHASE_ITERATIONS}
WITHDISPLAY = 'disp' in sys.argv

robot = loadTalosLegs()
robot.model.armature[6:] = .3
if WITHDISPLAY:
    robot.initDisplay(loadModel=True)

rmodel = robot.model
rdata = rmodel.createData()

# Setting up the 3d walking problem
rightFoot = 'right_sole_link'
leftFoot = 'left_sole_link'

rightId = rmodel.getFrameId(rightFoot)
leftId = rmodel.getFrameId(leftFoot)

# Create the initial state
q0 = rmodel.referenceConfigurations['half_sitting'].copy()
v0 = zero(rmodel.nv)
x0 = m2a(np.concatenate([q0, v0]))
rmodel.defaultState = x0.copy()

# Solving the 3d walking problem using DDP
stepLength = 0.2
swingDuration = 0.75
stanceDurantion = 0.1

dodisp = lambda xs, dt: displayTrajectory(robot, xs, dt)
disp = dodisp if WITHDISPLAY else lambda xs, dt: 0
disp.__defaults__ = (.1, )


def runningModel(contactIds, effectors, com=None, integrationStep=1e-2):
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
        contactModel.addContact('contact%d' % cid, ContactModel6D(rmodel, cid, ref=None))

    # Creating the cost model for a contact phase
    costModel = CostModelSum(rmodel, actModel.nu)
    wx = np.array([0] * 6 + [.1] * (rmodel.nv - 6) + [10] * rmodel.nv)
    costModel.addCost(
        'xreg',
        weight=1e-1,
        cost=CostModelState(
            rmodel, State, ref=rmodel.defaultState, nu=actModel.nu, activation=ActivationModelWeightedQuad(wx)))
    costModel.addCost('ureg', weight=1e-4, cost=CostModelControl(rmodel, nu=actModel.nu))
    for fid, ref in effectors.items():
        costModel.addCost("track%d" % fid, weight=100., cost=CostModelFramePlacement(rmodel, fid, ref, actModel.nu))

    if com is not None:
        costModel.addCost("com", weight=10000., cost=CostModelCoM(rmodel, ref=com, nu=actModel.nu))

    # Creating the action model for the KKT dynamics with simpletic Euler
    # integration scheme
    dmodel = DifferentialActionModelFloatingInContact(rmodel, actModel, contactModel, costModel)
    model = IntegratedActionModelEuler(dmodel)
    model.timeStep = integrationStep
    return model


def pseudoImpactModel(contactIds, effectors):
    # assert(len(effectors)==1)
    model = runningModel(contactIds, effectors, integrationStep=0)

    costModel = model.differential.costs
    for fid, ref in effectors.items():
        costModel.addCost('impactVel%d' % fid, weight=100., cost=CostModelFrameVelocity(rmodel, fid))
        costModel.costs['track%d' % fid].weight = 100
    costModel.costs['xreg'].weight = 1
    costModel.costs['ureg'].weight = 0.01

    return model


def impactModel(contactIds, effectors):
    State = StatePinocchio(rmodel)

    # Creating a 6D multi-contact model, and then including the supporting foot
    impulseModel = ImpulseModelMultiple(rmodel, {"impulse%d" % cid: ImpulseModel6D(rmodel, cid) for cid in contactIds})

    # Creating the cost model for a contact phase
    costModel = CostModelSum(rmodel, nu=0)
    wx = np.array([0] * 6 + [.1] * (rmodel.nv - 6) + [10] * rmodel.nv)
    costModel.addCost(
        'xreg',
        weight=.1,
        cost=CostModelState(rmodel, State, ref=rmodel.defaultState, nu=0, activation=ActivationModelWeightedQuad(wx)))
    costModel.addCost(
        'com', weight=1., cost=CostModelImpactCoM(rmodel, activation=ActivationModelWeightedQuad(m2a([.1, .1, 3.]))))
    for fid, ref in effectors.items():
        costModel.addCost("track%d" % fid, weight=100., cost=CostModelFramePlacement(rmodel, fid, ref, nu=0))
        # costModel.addCost("vel%d"%fid, weight=0.,
        #                   cost = CostModelFrameVelocity(rmodel,fid,nu=0))

    # Creating the action model for the KKT dynamics with simpletic Euler
    # integration scheme
    model = ActionModelImpact(rmodel, impulseModel, costModel)
    return model


# --- MODEL SEQUENCE
# --- MODEL SEQUENCE
# --- MODEL SEQUENCE
SE3 = pinocchio.SE3
pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
right0 = rdata.oMf[rightId].translation
left0 = rdata.oMf[leftId].translation
com0 = m2a(pinocchio.centerOfMass(rmodel, rdata, q0))

models = [runningModel([rightId, leftId], {}, integrationStep=4e-2) for i in range(10)]
models += [runningModel([], {}, integrationStep=5e-2) for i in range(6)]
models += [runningModel([], {}, com=com0 + [0, 0, 0.9], integrationStep=5e-2)]
models += [runningModel([], {}, integrationStep=5e-2) for i in range(8)]
models += [impactModel([leftId, rightId], {rightId: SE3(eye(3), right0), leftId: SE3(eye(3), left0)})]
models += [runningModel([rightId, leftId], {}, integrationStep=2e-2) for i in range(9)]
models += [runningModel([rightId, leftId], {}, integrationStep=0)]

high = [isinstance(m, IntegratedActionModelEuler) and 'com' in m.differential.costs.costs for m in models].index(True)
models[high].differential.costs['com'].cost.activation = ActivationModelInequality(
    np.array([-.01, -.01, -0.01]), np.array([.01, .01, 0.1]))

imp = [isinstance(m, ActionModelImpact) for m in models].index(True)
impact = models[imp]
impact.costs['track30'].weight = 0
impact.costs['track16'].weight = 0
impact.costs['com'].weight = 100
impact.costs['track16'].cost.activation = ActivationModelWeightedQuad(np.array([.2, 1, .1, 1, 1, 1]))
impact.costs['track30'].cost.activation = ActivationModelWeightedQuad(np.array([.2, 1, .1, 1, 1, 1]))
impact.costs.addCost(
    name='xycom',
    cost=CostModelCoM(rmodel, ref=com0, activation=ActivationModelWeightedQuad(np.array([1., .2, 0]))),
    weight=10)

for m in models[imp + 1:]:
    m.differential.costs['xreg'].weight = 0.0
    m.differential.contact['contact16'].gains[1] = 30
    m.differential.contact['contact30'].gains[1] = 30

models[-1].differential.costs['xreg'].weight = 1000
models[-1].differential.costs['xreg'].cost.activation.weights[:] = 1

# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------
# Solve only the take-off for an initial vanilla-flavor jump
# Solve the initial phase (take-off).
PHASE_NAME = "initial"
problem = ShootingProblem(initialState=x0, runningModels=models[:imp], terminalModel=models[imp])
ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPVerbose()]
ddp.th_stop = 1e-4
us0 = [
    m.differential.quasiStatic(d.differential, rmodel.defaultState) for m, d in zip(ddp.models(), ddp.datas())[:imp]
] + [np.zeros(0)] + [
    m.differential.quasiStatic(d.differential, rmodel.defaultState)
    for m, d in zip(ddp.models(), ddp.datas())[imp + 1:-1]
]

if PHASE_ITERATIONS[PHASE_NAME] > 0:
    print("*** SOLVE %s ***" % PHASE_NAME)
ddp.solve(
    maxiter=PHASE_ITERATIONS[PHASE_NAME],
    regInit=.1,
    init_xs=[rmodel.defaultState] * len(ddp.models()),
    init_us=us0[:imp])

if PHASE_ITERATIONS[PHASE_NAME] == 0:
    ddp.xs = [x for x in np.load(BACKUP_PATH + '%s.xs.npy' % PHASE_NAME)]
    ddp.us = [u for u in np.load(BACKUP_PATH + '%s.us.npy' % PHASE_NAME)]
elif PHASE_BACKUP[PHASE_NAME]:
    np.save(BACKUP_PATH + '%s.xs.npy' % PHASE_NAME, ddp.xs)
    np.save(BACKUP_PATH + '%s.us.npy' % PHASE_NAME, ddp.us)

# ---------------------------------------------------------------------------------------------
# Second phase of the search: optimize for increasing terminal angle of the waist.
PHASE_NAME = "angle"

models[imp].costs['xreg'].cost.activation.weights[3:6] = 100
impact.costs['xreg'].cost.activation.weights[3:rmodel.nv] = 10**6
ddp.th_stop = 5e-3

for ANG in np.arange(.5, 3.2, .3):
    models[imp].costs['xreg'].cost.ref[3:7] = [0, np.sin(ANG), 0, np.cos(ANG)]
    if PHASE_ITERATIONS[PHASE_NAME] > 0: print("*** SOLVE %s ang=%.1f ***" % (PHASE_NAME, ANG))
    ddp.solve(maxiter=PHASE_ITERATIONS[PHASE_NAME], regInit=.1, init_xs=ddp.xs, init_us=ddp.us, isFeasible=True)
    if PHASE_ITERATIONS[PHASE_NAME] > 0 and PHASE_BACKUP[PHASE_NAME]:
        np.save(BACKUP_PATH + '%s.%02d.xs.npy' % (PHASE_NAME, int(ANG * 10)), ddp.xs)
        np.save(BACKUP_PATH + '%s.%02d.us.npy' % (PHASE_NAME, int(ANG * 10)), ddp.us)

if PHASE_ITERATIONS[PHASE_NAME] == 0:
    ddp.xs = [x for x in np.load(BACKUP_PATH + '%s.xs.npy' % PHASE_NAME)]
    ddp.us = [u for u in np.load(BACKUP_PATH + '%s.us.npy' % PHASE_NAME)]
elif PHASE_BACKUP[PHASE_NAME]:
    np.save(BACKUP_PATH + '%s.xs.npy' % PHASE_NAME, ddp.xs)
    np.save(BACKUP_PATH + '%s.us.npy' % PHASE_NAME, ddp.us)

# ---------------------------------------------------------------------------------------------
# Third phase of the search: Solve both take-off and landing.
PHASE_NAME = "landing"
xsddp = ddp.xs
usddp = ddp.us

problem = ShootingProblem(initialState=x0, runningModels=models[:-1], terminalModel=models[-1])
ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPVerbose()]
ddp.th_stop = 1e-4

ddp.xs = xsddp + [rmodel.defaultState] * (len(models) - len(xsddp))
ddp.us = usddp + [
    np.zeros(0) if isinstance(m, ActionModelImpact) else m.differential.quasiStatic(
        d.differential, rmodel.defaultState) for m, d in zip(ddp.models(), ddp.datas())[len(usddp):-1]
]
impact.costs['track30'].weight = 1e6
impact.costs['track16'].weight = 1e6

if PHASE_ITERATIONS[PHASE_NAME] > 0: print("*** SOLVE %s ***" % PHASE_NAME)
ddp.solve(init_xs=ddp.xs, init_us=ddp.us, maxiter=PHASE_ITERATIONS[PHASE_NAME])

if PHASE_ITERATIONS[PHASE_NAME] == 0:
    ddp.xs = [x for x in np.load(BACKUP_PATH + '%s.xs.npy' % PHASE_NAME)]
    ddp.us = [u for u in np.load(BACKUP_PATH + '%s.us.npy' % PHASE_NAME)]
elif PHASE_BACKUP[PHASE_NAME]:
    np.save(BACKUP_PATH + '%s.xs.npy' % PHASE_NAME, ddp.xs)
    np.save(BACKUP_PATH + '%s.us.npy' % PHASE_NAME, ddp.us)

disp(ddp.xs)
