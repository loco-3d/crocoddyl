import numpy as np

import pinocchio
import crocoddyl
import example_robot_data
from test_utils import NUMDIFF_MODIFIER, assertNumDiff

# Loading ANYmal robot
# -----------------------------------------------------------------------------
robot = example_robot_data.load('anymal')
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
v = pinocchio.utils.rand(rmodel.nv)
x = np.concatenate([q, v])
u = pinocchio.utils.rand(rmodel.nv - 6)
# -------------------------------------------------

np.set_printoptions(linewidth=400, suppress=True)

state = crocoddyl.StateMultibody(rmodel)
actModel = crocoddyl.ActuationModelFloatingBase(state)
gains = pinocchio.utils.rand(2)
Mref_lf = crocoddyl.FramePlacement(rmodel.getFrameId('LF_FOOT'), pinocchio.SE3.Random())

contactModel6 = crocoddyl.ContactModel6D(state, Mref_lf, actModel.nu, gains)
rmodel.frames[Mref_lf.id].placement = pinocchio.SE3.Random()
contactModel = crocoddyl.ContactModelMultiple(state, actModel.nu)
contactModel.addContact("LF_FOOT_contact", contactModel6)

contactData = contactModel.createData(rdata)

model = crocoddyl.DifferentialActionModelContactFwdDynamics(state, actModel, contactModel,
                                                            crocoddyl.CostModelSum(state, actModel.nu), 0., True)

data = model.createData()

model.calc(data, x, u)
model.calcDiff(data, x, u)

mnum = crocoddyl.DifferentialActionModelNumDiff(model, False)
dnum = mnum.createData()
mnum.calc(dnum, x, u)
mnum.calcDiff(dnum, x, u)

model.costs.addCost(
    "momentum",
    crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelCentroidalMomentum(state, np.random.rand(6),
                                                                                 actModel.nu)), 1.)

data = model.createData()

cmodel = model.costs.costs['momentum'].cost
cdata = data.costs.costs['momentum']

model.calc(data, x, u)
model.calcDiff(data, x, u)

mnum = crocoddyl.DifferentialActionModelNumDiff(model, False)
dnum = mnum.createData()

mnum.calc(dnum, x, u)
mnum.calcDiff(dnum, x, u)
assertNumDiff(data.Fx, dnum.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu, NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (se
assertNumDiff(data.Lx, dnum.Lx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lu, dnum.Lu, NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 7e-3, is now 2.11e-4 (se
