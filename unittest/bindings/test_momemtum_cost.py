import numpy as np

import pinocchio
from crocoddyl import (ContactModel6D, ContactModelMultiple, CostModelSum, DifferentialActionModelNumDiff)
from crocoddyl import ActuationModelFloatingBase as ActuationModelFreeFloating
from crocoddyl import CostModelCentroidalMomentum as CostModelMomentum
from crocoddyl import DifferentialActionModelContactFwdDynamics as DifferentialActionModelFloatingInContact
from crocoddyl import StateMultibody as StatePinocchio
from crocoddyl import FramePlacement
from example_robot_data import loadANYmal
from crocoddyl.utils import a2m, m2a

from pinocchio.utils import rand
from test_utils import NUMDIFF_MODIFIER, assertNumDiff


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
# -----------------------------------------

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv - 6))
# -------------------------------------------------

np.set_printoptions(linewidth=400, suppress=True)

State = StatePinocchio(rmodel)
actModel = ActuationModelFreeFloating(State)
gains = pinocchio.utils.rand(2)
Mref_lf = FramePlacement(rmodel.getFrameId('LF_FOOT'), pinocchio.SE3.Random())

contactModel6 = ContactModel6D(State, Mref_lf, actModel.nu, gains)
rmodel.frames[Mref_lf.frame].placement = pinocchio.SE3.Random()
contactModel = ContactModelMultiple(State, actModel.nu)
contactModel.addContact("LF_FOOT_contact", contactModel6)

contactData = contactModel.createData(rdata)

model = DifferentialActionModelFloatingInContact(State, actModel, contactModel,
                                                 CostModelSum(State, actModel.nu, False), 0., True)

data = model.createData()

model.calc(data, x, u)
model.calcDiff(data, x, u)

mnum = DifferentialActionModelNumDiff(model, False)
dnum = mnum.createData()
mnum.calcDiff(dnum, x, u)

model.costs.addCost("momentum", CostModelMomentum(State, a2m(np.random.rand(6)), actModel.nu), 1.)

data = model.createData()

cmodel = model.costs.costs['momentum'].cost
cdata = data.costs.costs['momentum']

model.calcDiff(data, x, u)

mnum = DifferentialActionModelNumDiff(model, False)
dnum = mnum.createData()

mnum.calcDiff(dnum, x, u)
assertNumDiff(data.Fx, dnum.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu, NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (se
assertNumDiff(data.Lx, dnum.Lx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lu, dnum.Lu, NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (se
