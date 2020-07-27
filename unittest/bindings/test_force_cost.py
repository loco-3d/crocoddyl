import crocoddyl
import example_robot_data
import pinocchio
from test_utils import NUMDIFF_MODIFIER, assertNumDiff

# Create robot model and data
ROBOT_MODEL = example_robot_data.loadICub().model
ROBOT_DATA = ROBOT_MODEL.createData()
ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)

# Contact force cost
# Create differential action model and data; link the contact data
ACTUATION = crocoddyl.ActuationModelFloatingBase(ROBOT_STATE)
CONTACTS = crocoddyl.ContactModelMultiple(ROBOT_STATE, ACTUATION.nu)
CONTACT_6D_1 = crocoddyl.ContactModel6D(
    ROBOT_STATE, crocoddyl.FramePlacement(ROBOT_MODEL.getFrameId('r_sole'), pinocchio.SE3.Random()), ACTUATION.nu,
    pinocchio.utils.rand(2))
CONTACT_3D_2 = crocoddyl.ContactModel3D(
    ROBOT_STATE, crocoddyl.FrameTranslation(ROBOT_MODEL.getFrameId('l_sole'),
                                            pinocchio.SE3.Random().translation), ACTUATION.nu, pinocchio.utils.rand(2))
CONTACTS.addContact("r_sole_contact", CONTACT_6D_1)
CONTACTS.addContact("l_sole_contact", CONTACT_3D_2)
COSTS = crocoddyl.CostModelSum(ROBOT_STATE, ACTUATION.nu)
COSTS.addCost(
    "force_6d",
    crocoddyl.CostModelContactForce(ROBOT_STATE,
                                    crocoddyl.FrameForce(ROBOT_MODEL.getFrameId('r_sole'), pinocchio.Force.Random()),
                                    6, ACTUATION.nu), 1.)
COSTS.addCost(
    "force_3d",
    crocoddyl.CostModelContactForce(ROBOT_STATE,
                                    crocoddyl.FrameForce(ROBOT_MODEL.getFrameId('l_sole'), pinocchio.Force.Random()),
                                    3, ACTUATION.nu), 1.)
MODEL = crocoddyl.DifferentialActionModelContactFwdDynamics(ROBOT_STATE, ACTUATION, CONTACTS, COSTS, 0., True)
DATA = MODEL.createData()

# Created DAM numdiff
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

# Contact impulse cost
IMPULSES = crocoddyl.ImpulseModelMultiple(ROBOT_STATE)
IMPULSE_6D_1 = crocoddyl.ImpulseModel6D(ROBOT_STATE, ROBOT_MODEL.getFrameId('r_sole'))
IMPULSE_3D_2 = crocoddyl.ImpulseModel3D(ROBOT_STATE, ROBOT_MODEL.getFrameId('l_sole'))
IMPULSES.addImpulse("r_sole_contact", IMPULSE_6D_1)
IMPULSES.addImpulse("l_sole_contact", IMPULSE_3D_2)
COSTS = crocoddyl.CostModelSum(ROBOT_STATE, 0)
COSTS.addCost(
    "impulse_6d",
    crocoddyl.CostModelContactImpulse(ROBOT_STATE,
                                      crocoddyl.FrameForce(ROBOT_MODEL.getFrameId('r_sole'), pinocchio.Force.Random()),
                                      6), 1.)
COSTS.addCost(
    "impulse_3d",
    crocoddyl.CostModelContactImpulse(ROBOT_STATE,
                                      crocoddyl.FrameForce(ROBOT_MODEL.getFrameId('l_sole'), pinocchio.Force.Random()),
                                      3), 1.)
MODEL = crocoddyl.ActionModelImpulseFwdDynamics(ROBOT_STATE, IMPULSES, COSTS, 0., 0., True)
DATA = MODEL.createData()

# Created DAM numdiff
MODEL_ND = crocoddyl.ActionModelNumDiff(MODEL)
MODEL_ND.disturbance *= 10
dnum = MODEL_ND.createData()

x = ROBOT_STATE.rand()
u = pinocchio.utils.rand(0)
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
