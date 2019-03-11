import sys

import numpy as np
import pinocchio
from crocoddyl import (ActionModelImpact, ActivationModelWeightedQuad, ActuationModelFreeFloating, CallbackDDPLogger,
                       CallbackDDPVerbose, CallbackSolverDisplay, ContactModel6D, ContactModelMultiple, CostModelCoM,
                       CostModelControl, CostModelFramePlacement, CostModelState, CostModelSum,
                       DifferentialActionModelFloatingInContact, ImpulseModel6D, ImpulseModelMultiple,
                       IntegratedActionModelEuler, ShootingProblem, SolverDDP, StatePinocchio, a2m, loadTalosLegs, m2a)
from crocoddyl.diagnostic import displayTrajectory
from crocoddyl.fddp import SolverFDDP
from numpy.linalg import pinv
from pinocchio.utils import eye, zero

BACKUP_PATH = "npydata/jump."
WITHDISPLAY = 'disp' in sys.argv

robot = loadTalosLegs()
robot.model.armature[6:] = .3
if WITHDISPLAY:
    robot.initDisplay(loadModel=True)

rmodel = robot.model
rdata = rmodel.createData()

# Setting up the 3d walking problem
rightId = rmodel.getFrameId('right_sole_link')
leftId = rmodel.getFrameId('left_sole_link')

# Create the initial state
q0 = rmodel.referenceConfigurations['half_sitting'].copy()
v0 = zero(rmodel.nv)
x0 = m2a(np.concatenate([q0, v0]))
rmodel.defaultState = x0.copy()

# Solving the 3d walking problem using DDP
stepLength = 0.4
swingDuration = 0.75
stanceDurantion = 0.1

dodisp = lambda xs, dt: displayTrajectory(robot, xs, dt)
disp = dodisp if WITHDISPLAY else lambda xs, dt: 0
disp.__defaults__ = (.1, )


def runningModel(contactIds, effectors, com=None, integrationStep=1e-2):
    '''
    Creating the action model for floating-base systems. A walker system
    is by default a floating-base system.

    :params contactIds: list of frame Ids of points that should be in contact.
    :params effectors:  dict of key frame ids and SE3 values of effector references.
    :params com: if not None, com should be a array of size 3 used as a target com value.
    :params integrationStep: duration of the integration step in seconds.
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


def impactModel(contactIds, effectors):
    '''
    Creating the action model for floating-base systems during the impact phase.

    :params contactIds: list of frame Ids of points that should be in contact.
    :params effectors:  dict of key frame ids and SE3 values of effector references. This
    value should typically be provided for effector landing.
    '''
    State = StatePinocchio(rmodel)

    # Creating a 6D multi-contact model, and then including the supporting foot
    impulseModel = ImpulseModelMultiple(rmodel, {"impulse%d" % cid: ImpulseModel6D(rmodel, cid) for cid in contactIds})

    # Creating the cost model for a contact phase
    costModel = CostModelSum(rmodel, nu=0)
    wx = np.array([0] * 6 + [.1] * (rmodel.nv - 6) + [10] * rmodel.nv)
    costModel.addCost(
        'xreg',
        weight=1e-1,
        cost=CostModelState(rmodel, State, ref=rmodel.defaultState, nu=0, activation=ActivationModelWeightedQuad(wx)))
    # costModel.addCost('com',weight=1.,
    #                   cost=CostModelImpactCoM(rmodel,
    #                                           activation=ActivationModelWeightedQuad(m2a([.1,.1,3.]))))
    for fid, ref in effectors.items():
        wp = np.array([1.] * 6)
        costModel.addCost(
            "track%d" % fid,
            weight=1e5,
            cost=CostModelFramePlacement(rmodel, fid, ref, nu=0, activation=ActivationModelWeightedQuad(wp)))
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
right0 = m2a(rdata.oMf[rightId].translation)
left0 = m2a(rdata.oMf[leftId].translation)
com0 = m2a(pinocchio.centerOfMass(rmodel, rdata, q0))

KT = 3
KS = 8
models = [runningModel([rightId, leftId], {}, integrationStep=stanceDurantion / KT) for i in range(KT)]
models += [runningModel([rightId], {}, integrationStep=swingDuration / KS) for i in range(KS)]
models += [impactModel([leftId, rightId], {leftId: SE3(eye(3), a2m(left0 + [stepLength, 0, 0]))})]
models += [runningModel([rightId, leftId], {}, integrationStep=stanceDurantion / KT) for i in range(KT)]
models += [runningModel([leftId], {}, integrationStep=swingDuration / KS) for i in range(KS)]
models += [impactModel([leftId, rightId], {rightId: SE3(eye(3), a2m(right0 + [stepLength, 0, 0]))})]
models += [runningModel([rightId, leftId], {}, integrationStep=stanceDurantion / KT) for i in range(KT)]

imp1 = KT + KS
imp2 = 2 * (KT + KS) + 1
mimp1 = models[imp1]
mimp2 = models[imp2]

# ---- SAMPLE INIT VEL
if 'push' in sys.argv:
    from pinocchio.utils import zero
    vcom = np.matrix([3, 0., 0]).T
    pinocchio.computeAllTerms(rmodel, rdata, q0, v0)
    Jr = pinocchio.getFrameJacobian(rmodel, rdata, rightId, pinocchio.ReferenceFrame.LOCAL)
    Jl = pinocchio.getFrameJacobian(rmodel, rdata, leftId, pinocchio.ReferenceFrame.LOCAL)
    Jcom = rdata.Jcom

    v1 = pinv(np.vstack([Jr, Jl, Jcom])) * np.vstack([zero(12), vcom])
    x0[rmodel.nq:] = v1.flat

    mimp1.costs['track16'].cost.activation.weights[:2] = 0
    mimp2.costs['track30'].cost.activation.weights[:2] = 0

    for m in models:
        try:
            m.differential.contact['contact16'].gains[1] = 10
        except:
            pass
        try:
            m.differential.contact['contact30'].gains[1] = 10
        except:
            pass

# ---------------------------------------------------------------------------------------------
problem = ShootingProblem(initialState=x0, runningModels=models[:-1], terminalModel=models[-1])
ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPLogger(), CallbackDDPVerbose()]
if 'cb' in sys.argv and WITHDISPLAY:
    ddp.callback.append(CallbackSolverDisplay(robot, rate=-1))
ddp.th_stop = 1e-6

fddp = SolverFDDP(problem)
fddp.callback = [CallbackDDPLogger(), CallbackDDPVerbose()]
if 'cb' in sys.argv and WITHDISPLAY:
    fddp.callback.append(CallbackSolverDisplay(robot, rate=-1))
fddp.th_stop = 1e-6

us0 = [
    m.differential.quasiStatic(d.differential, rmodel.defaultState)
    if isinstance(m, IntegratedActionModelEuler) else np.zeros(0)
    for m, d in zip(ddp.problem.runningModels, ddp.problem.runningDatas)
]
xs0 = [rmodel.defaultState] * len(ddp.models())
xs1 = [problem.initialState] * len(ddp.models())
dimp1 = ddp.datas()[imp1]
dimp2 = ddp.datas()[imp2]

print("*** SOLVE ***")
fddp.solve(
    maxiter=50,
    # ,init_xs=xs0
    # ,init_xs=xs1
    # ,init_us=us0
)
