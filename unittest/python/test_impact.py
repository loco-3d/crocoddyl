import numpy as np
from numpy.linalg import norm

import pinocchio
from crocoddyl_legacy import (ActionModelImpact, ActionModelNumDiff, CostModelImpactCoM, CostModelSum, ImpulseModel6D,
                       ImpulseModelMultiple, a2m, loadTalosArm, loadTalosLegs)
from crocoddyl_legacy.impact import CostModelImpactWholeBody
from pinocchio.utils import zero
from testutils import NUMDIFF_MODIFIER, assertNumDiff, df_dq

pinocchio.switchToNumpyMatrix()

# --- TALOS ARM
robot = loadTalosArm(freeFloating=False)
rmodel = robot.model

opPointName = 'root_joint'
contactName = 'arm_left_7_joint'
contactName = 'gripper_left_fingertip_1_link'

# opPointName,contactName = contactName,opPointName
CONTACTFRAME = rmodel.getFrameId(contactName)
OPPOINTFRAME = rmodel.getFrameId(opPointName)

impulseModel = ImpulseModel6D(rmodel, rmodel.getFrameId(contactName))
costModel = CostModelImpactWholeBody(rmodel)
model = ActionModelImpact(rmodel, impulseModel, costModel)
model.impulseWeight = 1.
data = model.createData()

x = model.State.rand()
# x[7:13] = 0
q = a2m(x[:model.nq])
v = a2m(x[model.nq:])

model.calc(data, x)
model.calcDiff(data, x)

mnum = ActionModelNumDiff(model, withGaussApprox=True)
dnum = mnum.createData()

nx, ndx, nq, nv, nu = model.nx, model.ndx, model.nq, model.nv, model.nu

mnum.calcDiff(dnum, x, None)
assertNumDiff(dnum.Fx[:nv, :nv], data.Fx[:nv, :nv],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Fx[:nv, nv:], data.Fx[:nv, nv:],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Fx[nv:, nv:], data.Fx[nv:, nv:],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

rdata = rmodel.createData()

#              = -K^-1 [ M'(vnext-v) + J'^T f ]
#                      [ J' vnext             ]


# Check M'(vnext-v)
# Check M(vnext-v)
def Mdv(q, vnext, v):
    M = pinocchio.crba(rmodel, rdata, q)
    return M * (vnext - v)


dn = df_dq(rmodel, lambda _q: Mdv(_q, a2m(data.vnext), v), q)
g = rmodel.gravity
rmodel.gravity = pinocchio.Motion.Zero()
pinocchio.computeRNEADerivatives(rmodel, rdata, q, zero(rmodel.nv), a2m(data.vnext) - v)
d = rdata.dtau_dq.copy()
rmodel.gravity = g
assertNumDiff(d, dn,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-5, is now 2.11e-4 (see assertNumDiff.__doc__)

# Check J.T f
np.set_printoptions(precision=4, linewidth=200, suppress=True)


def Jtf(q, f):
    pinocchio.computeJointJacobians(rmodel, rdata, q)
    pinocchio.forwardKinematics(rmodel, rdata, q)
    pinocchio.updateFramePlacements(rmodel, rdata)
    J = pinocchio.getFrameJacobian(rmodel, rdata, CONTACTFRAME, pinocchio.ReferenceFrame.LOCAL)
    return J.T * f


dn = df_dq(rmodel, lambda _q: Jtf(_q, a2m(data.f)), q)
g = rmodel.gravity
rmodel.gravity = pinocchio.Motion.Zero()
pinocchio.computeRNEADerivatives(rmodel, rdata, q, zero(rmodel.nv), zero(rmodel.nv), data.impulse.forces)
d = rdata.dtau_dq.copy()
rmodel.gravity = g
assertNumDiff(d, -dn,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-5, is now 2.11e-4 (see assertNumDiff.__doc__)

# Check J.T f + M(vnext-v)
np.set_printoptions(precision=4, linewidth=200, suppress=True)


def MvJtf(q, vnext, v, f):
    M = pinocchio.crba(rmodel, rdata, q)
    pinocchio.computeJointJacobians(rmodel, rdata, q)
    pinocchio.forwardKinematics(rmodel, rdata, q)
    pinocchio.updateFramePlacements(rmodel, rdata)
    J = pinocchio.getFrameJacobian(rmodel, rdata, CONTACTFRAME, pinocchio.ReferenceFrame.LOCAL)
    return M * (vnext - v) - J.T * f


dn = df_dq(rmodel, lambda _q: MvJtf(_q, a2m(data.vnext), v, a2m(data.f)), q)
g = rmodel.gravity
rmodel.gravity = pinocchio.Motion.Zero()
pinocchio.computeRNEADerivatives(rmodel, rdata, q, zero(rmodel.nv), a2m(data.vnext) - v, data.impulse.forces)
d = rdata.dtau_dq.copy()
rmodel.gravity = g
assertNumDiff(d, dn,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-5, is now 2.11e-4 (see assertNumDiff.__doc__)


# Check J vnext
def Jv(q, vnext):
    pinocchio.computeJointJacobians(rmodel, rdata, q)
    pinocchio.forwardKinematics(rmodel, rdata, q)
    pinocchio.updateFramePlacements(rmodel, rdata)
    J = pinocchio.getFrameJacobian(rmodel, rdata, CONTACTFRAME, pinocchio.ReferenceFrame.LOCAL)
    return J * vnext


assertNumDiff(dnum.Fx[nv:, :nv], data.Fx[nv:, :nv],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(dnum.Fx, data.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Rx, data.Rx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Lx, data.Lx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assert (data.Fu.shape[1] == 0 and (data.Lu == 0 or data.Lu.shape == (0, )))

# --- TALOS LEGS
robot = loadTalosLegs()
rmodel = robot.model
rmodel.armature[6:] = 0.1

contactName = 'left_sole_link'
contactName = 'leg_right_6_joint'

CONTACTFRAME = rmodel.getFrameId(contactName)

impulse6 = ImpulseModel6D(rmodel, rmodel.getFrameId(contactName))
impulseModel = ImpulseModelMultiple(rmodel, {"6d": impulse6})
costModel = CostModelImpactWholeBody(rmodel)
model = ActionModelImpact(rmodel, impulse6, costModel)
data = model.createData()
model.impulseWeight = 1.

x = model.State.rand()
q = a2m(x[:model.nq])
v = a2m(x[model.nq:])

model.calc(data, x)
model.calcDiff(data, x)

rdata = rmodel.createData()

dn = df_dq(rmodel, lambda _q: Mdv(_q, a2m(data.vnext), v), q)
g = rmodel.gravity
rmodel.gravity = pinocchio.Motion.Zero()
pinocchio.computeRNEADerivatives(rmodel, rdata, q, zero(rmodel.nv), a2m(data.vnext) - v)
d = rdata.dtau_dq.copy()
rmodel.gravity = g
assertNumDiff(d, dn,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

# Check J.T f
np.set_printoptions(precision=4, linewidth=200, suppress=True)

dn = df_dq(rmodel, lambda _q: Jtf(_q, a2m(data.f)), q)
g = rmodel.gravity
rmodel.gravity = pinocchio.Motion.Zero()
pinocchio.computeRNEADerivatives(rmodel, rdata, q, zero(rmodel.nv), zero(rmodel.nv), data.impulse.forces)
d = -rdata.dtau_dq.copy()
rmodel.gravity = g
assertNumDiff(d, dn,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

# Check J.T f + M(vnext-v)
np.set_printoptions(precision=4, linewidth=200, suppress=True)

dn = df_dq(rmodel, lambda _q: MvJtf(_q, a2m(data.vnext), v, a2m(data.f)), q)
g = rmodel.gravity
rmodel.gravity = pinocchio.Motion.Zero()
pinocchio.computeRNEADerivatives(rmodel, rdata, q, zero(rmodel.nv), a2m(data.vnext) - v, data.impulse.forces)
d = rdata.dtau_dq.copy()
rmodel.gravity = g
assertNumDiff(d, dn,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)


# Check K'r-k' = [ M' (vnext-v) + J'f - M'v ]
#                [ J' vnext
def Krk(q, vnext, v, f):
    M = pinocchio.crba(rmodel, rdata, q)
    pinocchio.computeJointJacobians(rmodel, rdata, q)
    pinocchio.forwardKinematics(rmodel, rdata, q)
    pinocchio.updateFramePlacements(rmodel, rdata)
    J = pinocchio.getFrameJacobian(rmodel, rdata, CONTACTFRAME, pinocchio.ReferenceFrame.LOCAL)

    return np.vstack([M * (vnext - v) - J.T * f, J * vnext])


dn = df_dq(rmodel, lambda _q: Krk(_q, a2m(data.vnext), v, a2m(data.f)), q)
d = np.vstack([data.did_dq, data.dv_dq])
assertNumDiff(d, dn,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

dn = df_dq(rmodel, lambda _q: Krk(_q, a2m(data.vnext), v, a2m(data.f)), q)
d = np.vstack([data.did_dq, data.dv_dq])
assert (norm(d - dn) < 1e-4)

mnum = ActionModelNumDiff(model, withGaussApprox=True)
dnum = mnum.createData()

nx, ndx, nq, nv, nu = model.nx, model.ndx, model.nq, model.nv, model.nu

mnum.calcDiff(dnum, x, None)
assertNumDiff(dnum.Fx[:nv, :nv], data.Fx[:nv, :nv],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Fx[:nv, nv:], data.Fx[:nv, nv:],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Fx[nv:, :nv], data.Fx[nv:, :nv],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 3e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Fx[nv:, nv:], data.Fx[nv:, nv:],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(dnum.Fx, data.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Rx, data.Rx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Lx, data.Lx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assert (data.Fu.shape[1] == 0 and (data.Lu == 0 or data.Lu.shape == (0, )))

# ----------------------------------------------------------------------
# --- CHECK WITH SUM OF COSTS ------------------------------------------
# ----------------------------------------------------------------------

model.costs = CostModelSum(rmodel, nu=0)
model.costs.addCost(cost=costModel, weight=1, name="impactwb")
data = model.createData()

model.calc(data, x)
model.calcDiff(data, x)

mnum = ActionModelNumDiff(model, withGaussApprox=True)
dnum = mnum.createData()

nx, ndx, nq, nv, nu = model.nx, model.ndx, model.nq, model.nv, model.nu

mnum.calcDiff(dnum, x, None)
assertNumDiff(dnum.Fx[:nv, :nv], data.Fx[:nv, :nv],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Fx[:nv, nv:], data.Fx[:nv, nv:],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Fx[nv:, :nv], data.Fx[nv:, :nv],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 3e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Fx[nv:, nv:], data.Fx[nv:, nv:],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(dnum.Fx, data.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Rx, data.Rx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Lx, data.Lx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assert (data.Fu.shape[1] == 0 and (data.Lu == 0 or data.Lu.shape == (0, )))

costCom = CostModelImpactCoM(rmodel)
# model.costs.addCost( cost=costCom,weight=1,name="impactcom" )
model.costs = costCom

data = model.createData()

model.calc(data, x)
model.calcDiff(data, x)

mnum = ActionModelNumDiff(model, withGaussApprox=True)
dnum = mnum.createData()

nx, ndx, nq, nv, nu = model.nx, model.ndx, model.nq, model.nv, model.nu

mnum.calcDiff(dnum, x, None)
assertNumDiff(dnum.Fx[:nv, :nv], data.Fx[:nv, :nv],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Fx[:nv, nv:], data.Fx[:nv, nv:],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Fx[nv:, :nv], data.Fx[nv:, :nv],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 3e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Fx[nv:, nv:], data.Fx[nv:, nv:],
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

assertNumDiff(dnum.Fx, data.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Rx, data.Rx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Lx, data.Lx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assert (data.Fu.shape[1] == 0 and (data.Lu == 0 or data.Lu.shape == (0, )))
