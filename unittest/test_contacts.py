import numpy as np
from numpy.linalg import eig, norm, pinv

import pinocchio
from crocoddyl import (ActionModelNumDiff, ContactModel3D,
                       ContactModel6D, ContactModelMultiple, CostModelControl,
                       CostModelFrameTranslation, CostModelState, CostModelSum,
                       DifferentialActionModelNumDiff,
                       IntegratedActionModelEuler, ShootingProblem, SolverDDP)
from crocoddyl import ActuationModelFloatingBase as ActuationModelFreeFloating
from crocoddyl import CostModelContactForce as CostModelForce
from crocoddyl import DifferentialActionModelContactFwdDynamics as DifferentialActionModelFloatingInContact
from crocoddyl import StateMultibody as StatePinocchio
from crocoddyl import FramePlacement, FrameTranslation
from example_robot_data import loadANYmal
from crocoddyl.utils import a2m, m2a

from pinocchio.utils import rand, zero
from testutils import NUMDIFF_MODIFIER, assertNumDiff, df_dq, df_dx, EPS

def absmax(A):
    return np.max(abs(A))
    

# Loading Talos arm with FF TODO use a bided or quadruped
# -----------------------------------------------------------------------------
robot = loadANYmal()
robot.model.armature[6:] = 1.
qmin = robot.model.lowerPositionLimit
qmin[:7] = -1
robot.model.lowerPositionLimit = qmin
qmax = robot.model.upperPositionLimit
qmax[:7] = 1
robot.model.upperPositionLimit = qmax
rmodel = robot.model
rdata = rmodel.createData()

np.set_printoptions(linewidth=400, suppress=True)

State = StatePinocchio(rmodel)
actModel = ActuationModelFreeFloating(State)
gains = pinocchio.utils.rand(2)
Mref_lf = FramePlacement (rmodel.getFrameId('LF_FOOT'),
                          pinocchio.SE3.Random())
contactModel = ContactModel6D(State,
                              Mref_lf, actModel.nu,
                              gains)
contactData = contactModel.createData(rdata)

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv - 6))

pinocchio.forwardKinematics(rmodel, rdata, q, v, zero(rmodel.nv))
pinocchio.computeJointJacobians(rmodel, rdata)
pinocchio.updateFramePlacements(rmodel, rdata)
pinocchio.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, zero(rmodel.nv))
contactModel.calc(contactData, x)
contactModel.calcDiff(contactData, x)

rdata2 = rmodel.createData()
pinocchio.computeAllTerms(rmodel, rdata2, q, v)
pinocchio.updateFramePlacements(rmodel, rdata2)
contactData2 = contactModel.createData(rdata2)
contactModel.calc(contactData2, x)
assert (norm(contactData.a0 - contactData2.a0) < 1e-9)
assert (norm(contactData.Jc - contactData2.Jc) < 1e-9)


def returna_at0(q, v):
    x = np.vstack([q, v])
    pinocchio.computeAllTerms(rmodel, rdata2, q, v)
    pinocchio.updateFramePlacements(rmodel, rdata2)
    contactModel.calc(contactData2, x)
    return contactData2.a0.copy()


Aq_numdiff = df_dq(rmodel, lambda _q: returna_at0(_q, v), q)
Av_numdiff = df_dx(lambda _v: returna_at0(q, _v), v)

assertNumDiff(contactData.da0_dx[:, :rmodel.nv], Aq_numdiff,
              NUMDIFF_MODIFIER * np.sqrt(2 * EPS))  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(contactData.da0_dx[:, rmodel.nv:], Av_numdiff,
              NUMDIFF_MODIFIER * np.sqrt(2 * EPS))  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

eps = 1e-8
Aq_numdiff = df_dq(rmodel, lambda _q: returna_at0(_q, v), q, h=eps)
Av_numdiff = df_dx(lambda _v: returna_at0(q, _v), v, h=eps)

assert (np.isclose(contactData.da0_dx[:, :rmodel.nv], Aq_numdiff, atol=np.sqrt(eps)).all())
assert (np.isclose(contactData.da0_dx[:, rmodel.nv:], Av_numdiff, atol=np.sqrt(eps)).all())

Mref_lf_t = FrameTranslation (rmodel.getFrameId('LF_FOOT'),
                            pinocchio.SE3.Random().translation)

contactModel = ContactModel3D(State,
                              Mref_lf_t, actModel.nu,
                              gains)
contactData = contactModel.createData(rdata)

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv - 6))

pinocchio.forwardKinematics(rmodel, rdata, q, v, zero(rmodel.nv))
pinocchio.computeJointJacobians(rmodel, rdata)
pinocchio.updateFramePlacements(rmodel, rdata)
pinocchio.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, zero(rmodel.nv))
contactModel.calc(contactData, x)
contactModel.calcDiff(contactData, x)

rdata2 = rmodel.createData()
pinocchio.computeAllTerms(rmodel, rdata2, q, v)
pinocchio.updateFramePlacements(rmodel, rdata2)
contactData2 = contactModel.createData(rdata2)
contactModel.calc(contactData2, x)
assert (norm(contactData.a0 - contactData2.a0) < 1e-9)
assert (norm(contactData.Jc - contactData2.Jc) < 1e-9)


def returna0(q, v):
    x = np.vstack([q, v])
    pinocchio.computeAllTerms(rmodel, rdata2, q, v)
    pinocchio.updateFramePlacements(rmodel, rdata2)
    contactModel.calc(contactData2, x)
    return contactData2.a0.copy()


Aq_numdiff = df_dq(rmodel, lambda _q: returna0(_q, v), q)
Av_numdiff = df_dx(lambda _v: returna0(q, _v), v)

assertNumDiff(contactData.da0_dx[:, :rmodel.nv], Aq_numdiff,
              NUMDIFF_MODIFIER * np.sqrt(2 * EPS))  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(contactData.da0_dx[:, rmodel.nv:], Av_numdiff,
              NUMDIFF_MODIFIER * np.sqrt(2 * EPS))  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

Aq_numdiff = df_dq(rmodel, lambda _q: returna0(_q, v), q, h=eps)
Av_numdiff = df_dx(lambda _v: returna0(q, _v), v, h=eps)

assert (np.isclose(contactData.da0_dx[:, :rmodel.nv], Aq_numdiff, atol=np.sqrt(eps)).all())
assert (np.isclose(contactData.da0_dx[:, rmodel.nv:], Av_numdiff, atol=np.sqrt(eps)).all())

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv) * 2 - 1

pinocchio.computeJointJacobians(rmodel, rdata, q)
J6 = pinocchio.getJointJacobian(rmodel, rdata, rmodel.joints[-1].id, pinocchio.ReferenceFrame.LOCAL).copy()
J = J6[:3, :]
v -= pinv(J) * J * v

x = np.vstack([q, v])
u = a2m(np.random.rand(rmodel.nv - 6) * 2 - 1)

actModel = ActuationModelFreeFloating(State)
contactModel3 = ContactModel3D(State,
                               Mref_lf_t, actModel.nu, gains)

rmodel.frames[Mref_lf_t.frame].placement = pinocchio.SE3.Random()
contactModel = ContactModelMultiple(State, actModel.nu)
contactModel.addContact("LF_FOOT_contact",contactModel3)

model = DifferentialActionModelFloatingInContact(State,
                                                 actModel, contactModel,
                                                 CostModelSum(State, actModel.nu, False), 0.,
                                                 True)
data = model.createData()

model.calc(data, x, u)
#assert (len(list(filter(lambda x: x > 0, eig(data.K)[0]))) == model.nv)
#assert (len(list(filter(lambda x: x < 0, eig(data.K)[0]))) == model.ncontact)
_taucheck = pinocchio.rnea(rmodel, rdata, q, v, data.xout, data.contacts.fext)
#_taucheck.flat[:] += rmodel.armature.flat * data.a
assert (absmax(_taucheck[:6]) < 1e-6)
assert (absmax(m2a(_taucheck[6:] - u)) < 1e-6)

model.calcDiff(data, x, u)

mnum = DifferentialActionModelNumDiff(model, False)
dnum = mnum.createData()
mnum.calcDiff(dnum, x, u)
assertNumDiff(data.Fx, dnum.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

# ------------------------------------------------
contactModel6 = ContactModel6D(State, Mref_lf, actModel.nu, gains)
rmodel.frames[Mref_lf.frame].placement = pinocchio.SE3.Random()
contactModel = ContactModelMultiple(State, actModel.nu)
contactModel.addContact("LF_FOOT_contact", contactModel6)

model = DifferentialActionModelFloatingInContact(State,
                                                 actModel, contactModel,
                                                 CostModelSum(State, actModel.nu, False),
                                                 0.,True)
data = model.createData()

model.calc(data, x, u)
#assert (len(list(filter(lambda x: x > 0, eig(data.K)[0]))) == model.nv)
#assert (len(list(filter(lambda x: x < 0, eig(data.K)[0]))) == model.ncontact)
_taucheck = pinocchio.rnea(rmodel, rdata, q, v, data.xout, data.contacts.fext)
#_taucheck.flat[:] += rmodel.armature.flat * data.a
assert (absmax(_taucheck[:6]) < 1e-6)
assert (absmax(m2a(_taucheck[6:] - u)) < 1e-6)

model.calcDiff(data, x, u)

mnum = DifferentialActionModelNumDiff(model, False)
dnum = mnum.createData()
mnum.calcDiff(dnum, x, u)
assertNumDiff(data.Fx, dnum.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (see assertNumDiff.__doc__)


# ----------------------------------------------------------
# Check force derivatives
def calcForces(q_, v_, u_):
    model.calc(data, np.vstack([q_, v_]), u_)
    return data.pinocchio.lambda_c.copy()


Fq = df_dq(rmodel, lambda _q: calcForces(_q, v, u), q)
Fv = df_dx(lambda _v: calcForces(q, _v, u), v)
Fu = df_dx(lambda _u: calcForces(q, v, _u), u)
assertNumDiff(Fq, data.df_dx[:,:robot.nv], 1e-3)
#NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(Fv, data.df_dx[:,robot.nv:],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(Fu, data.df_du,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

Fq = df_dq(rmodel, lambda _q: calcForces(_q, v, u), q)
Fv = df_dx(lambda _v: calcForces(q, _v, u), v)
Fu = df_dx(lambda _u: calcForces(q, v, _u), u)
assert (absmax(Fq - data.df_dx[:,:rmodel.nv]) < 1e-3)
assert (absmax(Fv - data.df_dx[:,rmodel.nv:]) < 1e-3)
assert (absmax(Fu - data.df_du) < 1e-3)

#model.costs = CostModelSum(State, actModel.nu)
model.costs.addCost("force",
                    CostModelForce(State,
                                        model.contacts.contacts["LF_FOOT_contact"].contact,
                                        a2m(np.random.rand(6)), actModel.nu),
                    1.)

data = model.createData()
data.costs.costs["force"].contact = data.contacts.contacts["LF_FOOT_contact"]

cmodel = model.costs.costs['force'].cost
cdata = data.costs.costs['force']

model.calcDiff(data, x, u)

mnum = DifferentialActionModelNumDiff(model, False)
dnum = mnum.createData()
for d in dnum.data_x:
    d.costs.costs["force"].contact = d.contacts.contacts["LF_FOOT_contact"]
for d in dnum.data_u:
    d.costs.costs["force"].contact = d.contacts.contacts["LF_FOOT_contact"]
dnum.data_0.costs.costs["force"].contact = dnum.data_0.contacts.contacts["LF_FOOT_contact"]

mnum.calcDiff(dnum, x, u)
assertNumDiff(data.Fx, dnum.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

"""
nfaces = 10
nc = model.contact.ncontact
A = np.random.rand(nfaces, nc)

model.costs = CostModelSum(rmodel, actModel.nu, False)
model.costs.addCost(name='force_cone',
                    weight=1,
                    cost=CostModelForceLinearCone(rmodel, model.contact.contacts['fingertip'], A, nu=actModel.nu))
data = model.createData()
data.costs['force_cone'].contact = data.contact[model.costs['force_cone'].cost.contact]

cmodel = model.costs['force_cone'].cost
cdata = data.costs['force_cone']

model.calcDiff(data, x, u)

# Check derivative of the model.
mnum = DifferentialActionModelNumDiff(model, withGaussApprox=False)
dnum = mnum.createData()
for d in dnum.data_x:
    d.costs['force_cone'].contact = d.contact[model.costs['force_cone'].cost.contact]
for d in dnum.datau:
    d.costs['force_cone'].contact = d.contact[model.costs['force_cone'].cost.contact]
dnum.data0.costs['force_cone'].contact = dnum.data0.contact[model.costs['force_cone'].cost.contact]

mnum.calcDiff(dnum, x, u)
assertNumDiff(data.Fx, dnum.Fx,
              1e4 * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu,
              1e4 * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lx, dnum.Lx,
              1e6 * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-2 (see assertNumDiff.__doc__)
assertNumDiff(data.Lu, dnum.Lu,
              1e6 * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-2 (see assertNumDiff.__doc__)

assert (absmax(data.Lx - dnum.Lx) / model.nx < 1e-3)
assert (absmax(data.Lu - dnum.Lu) / model.nu < 1e-3)


actModel = ActuationModelFreeFloating(rmodel)
contactModel = ContactModelMultiple(rmodel)
contactModel.addContact(name='root_joint', contact=contactModel6)

costModel = CostModelSum(rmodel, actModel.nu, False)
costModel.addCost(name="pos",
                  weight=10,
                  cost=CostModelFrameTranslation(rmodel,
                                                 nu=actModel.nu,
                                                 frame=rmodel.getFrameId('gripper_left_inner_single_link'),
                                                 ref=np.array([.5, .4, .3])))
costModel.addCost(name="regx", weight=0.1, cost=CostModelState(rmodel, State, ref=State.zero(), nu=actModel.nu))
costModel.addCost(name="regu", weight=0.01, cost=CostModelControl(rmodel, nu=actModel.nu))

c1 = costModel.costs['pos'].cost
c2 = costModel.costs['regx'].cost
c3 = costModel.costs['regu'].cost

dmodel = DifferentialActionModelFloatingInContact(rmodel, actModel, contactModel,
                                                  costModel,
                                                  enable_force=True)
model = IntegratedActionModelEuler(dmodel)
data = model.createData()

d1 = data.differential.costs.costs['pos']
d2 = data.differential.costs.costs['regx']
d3 = data.differential.costs.costs['regu']

mnum = ActionModelNumDiff(model, withGaussApprox=True)
dnum = mnum.createData()

# This trigger an error x[3:7] = [1,0,0,0]
x[3:7] = [0, 0, 0, 1]  # TODO: remove this after adding assertion to include any case

model.calc(data, x, u)
model.calcDiff(data, x, u)
mnum.calcDiff(dnum, x, u)
assertNumDiff(data.Lx, dnum.Lx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lu, dnum.Lu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Lxx, data.Lxx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Lxu, data.Lxu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Luu, data.Luu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

# --- test quasi static guess
x0 = x.copy()
x0[dmodel.nq:] = 0
for c in dmodel.contact.contacts.values():
    c.gains = [0., 0.]
u0 = dmodel.quasiStatic(data.differential, x0)
a0, _ = dmodel.calc(data.differential, x0, u0)

assert (norm(a0) < 1e-6)

# TODO move to an integrative test
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# integrative test: checking 1-step DDP versus KKT

model.timeStep = 1e-1
dmodel.costs['pos'].weight = 1
dmodel.costs['regx'].weight = 0
dmodel.costs['regu'].weight = 0

# Choose a cost that is reachable.
x0 = model.State.rand()
xref = model.State.rand()
xref[:7] = x0[:7]
xref[3:7] = [0, 0, 0, 1]  # TODO: remove this after adding assertion to include any case
pinocchio.forwardKinematics(rmodel, rdata, a2m(xref[:rmodel.nq]))
pinocchio.updateFramePlacements(rmodel, rdata)
c1.ref[:] = m2a(rdata.oMf[c1.frame].translation.copy())

problem = ShootingProblem(x0, [model], model)

ddp = SolverDDP(problem)
ddp.callback = [CallbackDDPLogger()]
ddp.th_stop = 1e-18
xddp, uddp, doneddp = ddp.solve(maxiter=400)

assert (doneddp)
assert (norm(ddp.datas()[-1].differential.costs['pos'].residuals) < 1e-3)
assert (norm(m2a(ddp.datas()[-1].differential.costs['pos'].pinocchio.oMf[c1.frame].translation) - c1.ref) < 1e-3)

u0 = np.zeros(model.nu)
x1 = model.calc(data, problem.initialState, u0)[0]
x0s = [problem.initialState.copy(), x1]
u0s = [u0.copy()]

dmodel.costs['regu'].weight = 1e-3

kkt = SolverKKT(problem)
kkt.th_stop = 1e-18
xkkt, ukkt, donekkt = kkt.solve(init_xs=x0s, init_us=u0s, isFeasible=True, maxiter=20)
xddp, uddp, doneddp = ddp.solve(init_xs=x0s, init_us=u0s, isFeasible=True, maxiter=20)

assert (donekkt)
assert (norm(xkkt[0] - problem.initialState) < 1e-9)
assert (norm(xddp[0] - problem.initialState) < 1e-9)
for t in range(problem.T):
    assert (norm(ukkt[t] - uddp[t]) < 1e-6)
    assert (norm(xkkt[t + 1] - xddp[t + 1]) < 1e-6)
"""
