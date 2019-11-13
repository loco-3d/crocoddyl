import numpy as np
from numpy.linalg import norm, pinv

import pinocchio
from crocoddyl import (ContactModel3D, ContactModel6D, ContactModelMultiple, CostModelSum,
                       DifferentialActionModelNumDiff)
from crocoddyl import ActuationModelFloatingBase as ActuationModelFreeFloating
from crocoddyl import CostModelContactForce as CostModelForce
from crocoddyl import DifferentialActionModelContactFwdDynamics as DifferentialActionModelFloatingInContact
from crocoddyl import StateMultibody as StatePinocchio
from crocoddyl import FramePlacement, FrameTranslation
from example_robot_data import loadICub
from crocoddyl.utils import a2m, m2a

from pinocchio.utils import rand, zero
from testutils import NUMDIFF_MODIFIER, assertNumDiff, df_dq, df_dx, EPS
NUMDIFF_MODIFIER = 1e6

def absmax(A):
    return np.max(abs(A))


# Loading Talos arm with FF TODO use a bided or quadruped
# -----------------------------------------------------------------------------
# robot = loadANYmal()
robot = loadICub()
rmodel = robot.model
rdata = rmodel.createData()

np.set_printoptions(linewidth=400, suppress=True)

State = StatePinocchio(rmodel)
actModel = ActuationModelFreeFloating(State)
gains = pinocchio.utils.rand(2)
Mref_lf = FramePlacement(rmodel.getFrameId('r_sole'), pinocchio.SE3.Random())

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv - 6))

# ------------------------------------------------
contactModel6 = ContactModel6D(State, Mref_lf, actModel.nu, gains)
rmodel.frames[Mref_lf.frame].placement = pinocchio.SE3.Random()
contactModel = ContactModelMultiple(State, actModel.nu)
contactModel.addContact("r_sole_contact", contactModel6)

model = DifferentialActionModelFloatingInContact(State, actModel, contactModel,
                                                 CostModelSum(State, actModel.nu, False), 0., True)
data = model.createData()

model.calc(data, x, u)
# assert (len(list(filter(lambda x: x > 0, eig(data.K)[0]))) == model.nv)
# assert (len(list(filter(lambda x: x < 0, eig(data.K)[0]))) == model.ncontact)
_taucheck = pinocchio.rnea(rmodel, rdata, q, v, data.xout, data.contacts.fext)
# _taucheck.flat[:] += rmodel.armature.flat * data.a
assert (absmax(_taucheck[:6]) < 1e-6)
assert (absmax(m2a(_taucheck[6:]) - u) < 1e-6)

model.calcDiff(data, x, u)

mnum = DifferentialActionModelNumDiff(model, False)
dnum = mnum.createData()
mnum.calcDiff(dnum, x, u)
assertNumDiff(data.Fx, dnum.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

# model.costs = CostModelSum(State, actModel.nu)
model.costs.addCost(
    "force",
    CostModelForce(State, model.contacts.contacts["r_sole_contact"].contact, a2m(np.random.rand(6)), actModel.nu), 1.)

data = model.createData()
data.costs.costs["force"].contact = data.contacts.contacts["r_sole_contact"]

cmodel = model.costs.costs['force'].cost
cdata = data.costs.costs['force']

model.calcDiff(data, x, u)

mnum = DifferentialActionModelNumDiff(model, False)
dnum = mnum.createData()
for d in dnum.data_x:
    d.costs.costs["force"].contact = d.contacts.contacts["r_sole_contact"]
for d in dnum.data_u:
    d.costs.costs["force"].contact = d.contacts.contacts["r_sole_contact"]
dnum.data_0.costs.costs["force"].contact = dnum.data_0.contacts.contacts["r_sole_contact"]

mnum.calcDiff(dnum, x, u)
assertNumDiff(data.Fx, dnum.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lx, dnum.Lx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lu, dnum.Lu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
