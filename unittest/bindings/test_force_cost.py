import crocoddyl
import pinocchio
import example_robot_data
import numpy as np

crocoddyl.switchToNumpyMatrix()

# Create robot model and data
ROBOT_MODEL = example_robot_data.loadICub().model
ROBOT_DATA = ROBOT_MODEL.createData()

# Create differential action model and data; link the contact data
ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)
ACTUATION = crocoddyl.ActuationModelFloatingBase(ROBOT_STATE)
CONTACTS = crocoddyl.ContactModelMultiple(ROBOT_STATE, ACTUATION.nu)
CONTACT_6D_1 = crocoddyl.ContactModel6D(
    ROBOT_STATE, crocoddyl.FramePlacement(ROBOT_MODEL.getFrameId('r_sole'), pinocchio.SE3.Random()), ACTUATION.nu,
    pinocchio.utils.rand(2))
CONTACT_6D_2 = crocoddyl.ContactModel6D(
    ROBOT_STATE, crocoddyl.FramePlacement(ROBOT_MODEL.getFrameId('l_sole'), pinocchio.SE3.Random()), ACTUATION.nu,
    pinocchio.utils.rand(2))
CONTACTS.addContact("r_sole_contact", CONTACT_6D_1)
CONTACTS.addContact("l_sole_contact", CONTACT_6D_2)
COSTS = crocoddyl.CostModelSum(ROBOT_STATE, ACTUATION.nu, False)
COSTS.addCost(
    "force",
    crocoddyl.CostModelContactForce(ROBOT_STATE,
                                    crocoddyl.FrameForce(ROBOT_MODEL.getFrameId('r_sole'), pinocchio.Force.Random()),
                                    ACTUATION.nu), 1.)
MODEL = crocoddyl.DifferentialActionModelContactFwdDynamics(ROBOT_STATE, ACTUATION, CONTACTS, COSTS, 0., True)
DATA = MODEL.createData()

# Created DAM numdiff
MODEL_ND = crocoddyl.DifferentialActionModelNumDiff(MODEL)
dnum = MODEL_ND.createData()

x = ROBOT_STATE.rand()
u = pinocchio.utils.rand(ACTUATION.nu)
MODEL.calcDiff(DATA, x, u)
MODEL_ND.calcDiff(dnum, x, u)
assert (np.allclose(DATA.Fx, dnum.Fx, atol=1e5 * MODEL_ND.disturbance))
assert (np.allclose(DATA.Fu, dnum.Fu, atol=1e5 * MODEL_ND.disturbance))
assert (np.allclose(DATA.Lx, dnum.Lx, atol=1e5 * MODEL_ND.disturbance))
assert (np.allclose(DATA.Lu, dnum.Lu, atol=1e5 * MODEL_ND.disturbance))
