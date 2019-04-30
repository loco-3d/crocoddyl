'''
This is an integrative test that is validating the feasibility part of the ddp solver
when armature is low.
In the test, we are setting up a DDP problem for the biped with DAMFIC models, imposing a
low (0.01) armature and running one step of the solver. The test is validated if the cost is
reasonable after the end of the search.

This test is not super strong. The low-armature model was already validated in the unittest of
DAMManipulator and DAMFIC, so we already knew that the model and its derivatives were correct.
The only "new" functionality to be tested here is the use of low-armature in the solver.
Any stronger test here is welcome.
'''
import numpy as np
import pinocchio
from crocoddyl import (ActivationModelWeightedQuad, ActuationModelFreeFloating, ContactModel6D, ContactModelMultiple,
                       CostModelCoM, CostModelControl, CostModelFramePlacement, CostModelState, CostModelSum,
                       DifferentialActionModelFloatingInContact, IntegratedActionModelEuler, ShootingProblem,
                       SolverDDP, StatePinocchio, a2m, loadTalosLegs, m2a)
from pinocchio.utils import eye, zero

robot = loadTalosLegs()
robot.model.armature[6:] = .01

rmodel = robot.model
rdata = rmodel.createData()
rmodel.q0 = rmodel.referenceConfigurations['half_sitting'].copy()

# Setting up the 3d walking problem
rightFoot = 'right_sole_link'
leftFoot = 'left_sole_link'

rightId = rmodel.getFrameId(rightFoot)
leftId = rmodel.getFrameId(leftFoot)

# Create the initial state
q0 = rmodel.q0
v0 = zero(rmodel.nv)
x0 = m2a(np.concatenate([q0, v0]))
rmodel.defaultState = x0.copy()


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
    costModel.addCost('xreg',
                      weight=1e-1,
                      cost=CostModelState(rmodel,
                                          State,
                                          ref=rmodel.defaultState,
                                          nu=actModel.nu,
                                          activation=ActivationModelWeightedQuad(wx)))
    costModel.addCost('ureg', weight=1e-4, cost=CostModelControl(rmodel, nu=actModel.nu))
    for fid, ref in effectors.items():
        if not isinstance(ref, SE3):
            ref = SE3(eye(3), a2m(ref))
        costModel.addCost("track%d" % fid, weight=100., cost=CostModelFramePlacement(rmodel, fid, ref, actModel.nu))

    if com is not None:
        costModel.addCost("com", weight=100., cost=CostModelCoM(rmodel, ref=com, nu=actModel.nu))

    # Creating the action model for the KKT dynamics with simpletic Euler
    # integration scheme
    dmodel = DifferentialActionModelFloatingInContact(rmodel, actModel, contactModel, costModel)
    model = IntegratedActionModelEuler(dmodel)
    model.timeStep = integrationStep
    return model


# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------

q = rmodel.q0.copy()
v = zero(rmodel.nv)
x = m2a(np.concatenate([q, v]))

SE3 = pinocchio.SE3
pinocchio.forwardKinematics(rmodel, rdata, q0)
pinocchio.updateFramePlacements(rmodel, rdata)
right0 = m2a(rdata.oMf[rightId].translation)
left0 = m2a(rdata.oMf[leftId].translation)
com0 = m2a(pinocchio.centerOfMass(rmodel, rdata, q0))

models = []
models += [runningModel([rightId, leftId], {}, integrationStep=5e-2)] * 10
models += [runningModel([rightId, leftId], {}, com=com0 + [0.1, 0, 0], integrationStep=5e-2)]
models += [runningModel([rightId], {}, integrationStep=5e-2)] * 10
models += [runningModel([rightId], {leftId: left0 + [0, 0, 0.1]}, com=com0 + [-0.1, 0, 0], integrationStep=5e-2)]

pass1 = models[10]
pass2 = models[21]

pass1.differential.costs['com'].weight = 100000
pass2.differential.costs['com'].weight = 100000
pass2.differential.costs['track16'].weight = 100000

# --- DDP
problem = ShootingProblem(initialState=x0, runningModels=models[:-1], terminalModel=models[-1])
ddp = SolverDDP(problem)
# ddp.callback = [ CallbackDDPLogger(), CallbackDDPVerbose() ]
ddp.th_stop = 1e-6

ddp.setCandidate()
m = models[0]
d = m.createData()

m = m.differential
d = d.differential

# m=m.differential.contact['contact30']
# d=d.differential.contact['contact30']
m.calcDiff(d, ddp.xs[0], ddp.us[0])
# m.calc(d,ddp.xs[0])

ddp.solve(maxiter=1,
          regInit=.1,
          init_xs=[rmodel.defaultState] * len(ddp.models()),
          init_us=[
              _m.differential.quasiStatic(_d.differential, rmodel.defaultState)
              for _m, _d in zip(ddp.models(), ddp.datas())[:-1]
          ])

assert (ddp.cost < 1e5)
'''
# --- PLOT
np.set_printoptions(precision=4, linewidth=200, suppress=True)
import matplotlib.pylab as plt
plt.ion()
plt.plot([ d.differential.pinocchio.com[0][0,0] for d in ddp.datas() ])
'''
