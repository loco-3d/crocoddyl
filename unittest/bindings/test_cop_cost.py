import crocoddyl
import pinocchio
import example_robot_data
import numpy as np

from test_utils import NUMDIFF_MODIFIER, assertNumDiff

# Create robot model, data, state and actuation
ROBOT_MODEL = example_robot_data.loadICub().model
ROBOT_DATA = ROBOT_MODEL.createData()
ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)
ACTUATION = crocoddyl.ActuationModelFloatingBase(ROBOT_STATE)

# Contact CoP position cost unittest
CONTACTS = crocoddyl.ContactModelMultiple(ROBOT_STATE, ACTUATION.nu)
CONTACT_6D_RF = crocoddyl.ContactModel6D(
    ROBOT_STATE, crocoddyl.FramePlacement(ROBOT_MODEL.getFrameId('r_sole'), pinocchio.SE3.Random()), ACTUATION.nu,
    pinocchio.utils.rand(2))
CONTACT_6D_LF = crocoddyl.ContactModel6D(
    ROBOT_STATE, crocoddyl.FramePlacement(ROBOT_MODEL.getFrameId('l_sole'), pinocchio.SE3.Random()), ACTUATION.nu,
    pinocchio.utils.rand(2))
CONTACTS.addContact("r_sole_contact", CONTACT_6D_RF)
CONTACTS.addContact("l_sole_contact", CONTACT_6D_LF)
COSTS = crocoddyl.CostModelSum(ROBOT_STATE, ACTUATION.nu)
COSTS.addCost(
    "r_sole_cop",
    crocoddyl.CostModelContactCoPPosition(ROBOT_STATE, crocoddyl.ActivationModelQuad(4), # TODO: Hard coded dim?
    crocoddyl.FrameFootGeometry(ROBOT_MODEL.getFrameId('r_sole'), (20, 8)), 1.))
COSTS.addCost(
    "l_sole_cop",
    crocoddyl.CostModelContactCoPPosition(ROBOT_STATE, crocoddyl.ActivationModelQuad(4), 
    crocoddyl.FrameFootGeometry(ROBOT_MODEL.getFrameId('l_sole'), (20, 8)), 1.))
MODEL = crocoddyl.DifferentialActionModelContactFwdDynamics(ROBOT_STATE, ACTUATION, CONTACTS, COSTS, 0., True)
DATA = MODEL.createData()

MODEL_ND = crocoddyl.DifferentialActionModelNumDiff(MODEL)
MODEL_ND.disturbance *= 10
dnum = MODEL_ND.createData()

x = ROBOT_STATE.rand()
u = pinocchio.utils.rand(ACTUATION.nu)
MODEL.calc(DATA, x, u)
MODEL.calcDiff(DATA, x, u)
MODEL_ND.calc(dnum, x, u)
MODEL_ND.calcDiff(dnum, x, u)
assertNumDiff(DATA.Fx, dnum.Fx, NUMDIFF_MODIFIER *
              MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(DATA.Fu, dnum.Fu, NUMDIFF_MODIFIER *
              MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(DATA.Lx, dnum.Lx, NUMDIFF_MODIFIER *
              MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(DATA.Lu, dnum.Lu, NUMDIFF_MODIFIER *
              MODEL_ND.disturbance)  # threshold was 2.7e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
