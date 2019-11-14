import crocoddyl
import pinocchio
import example_robot_data
import numpy as np

# Create robot model and data
ROBOT_MODEL = example_robot_data.loadICub().model
ROBOT_DATA = ROBOT_MODEL.createData()

# Create differential action model and data; link the contact data
ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)
ACTUATION = crocoddyl.ActuationModelFloatingBase(ROBOT_STATE)
CONTACTS = crocoddyl.ContactModelMultiple(ROBOT_STATE, ACTUATION.nu)
CONTACT_6D = crocoddyl.ContactModel6D(
    ROBOT_STATE, crocoddyl.FramePlacement(ROBOT_MODEL.getFrameId('r_sole'), pinocchio.SE3.Random()), ACTUATION.nu,
    pinocchio.utils.rand(2))
CONTACTS.addContact("r_sole_contact", CONTACT_6D)
costs = crocoddyl.CostModelSum(ROBOT_STATE, ACTUATION.nu, False)
costs.addCost("force", crocoddyl.CostModelContactForce(ROBOT_STATE, CONTACT_6D, pinocchio.utils.rand(6), ACTUATION.nu),
              1.)
MODEL = crocoddyl.DifferentialActionModelContactFwdDynamics(ROBOT_STATE, ACTUATION, CONTACTS, costs, 0., True)
DATA = MODEL.createData()
DATA.costs.costs["force"].contact = DATA.contacts.contacts["r_sole_contact"]

# Created DAM numdiff
MODEL_ND = crocoddyl.DifferentialActionModelNumDiff(MODEL, False)
dnum = MODEL_ND.createData()
for d in dnum.data_x:
    d.costs.costs["force"].contact = d.contacts.contacts["r_sole_contact"]
for d in dnum.data_u:
    d.costs.costs["force"].contact = d.contacts.contacts["r_sole_contact"]
dnum.data_0.costs.costs["force"].contact = dnum.data_0.contacts.contacts["r_sole_contact"]

x = ROBOT_STATE.rand()
u = pinocchio.utils.rand(ACTUATION.nu)
MODEL.calcDiff(DATA, x, u)
MODEL_ND.calcDiff(dnum, x, u)
np.allclose(DATA.Fx, dnum.Fx, atol=1e6 * MODEL_ND.disturbance)
np.allclose(DATA.Fu, dnum.Fu, atol=1e6 * MODEL_ND.disturbance)
np.allclose(DATA.Lx, dnum.Lx, atol=1e6 * MODEL_ND.disturbance)
np.allclose(DATA.Lu, dnum.Lu, atol=1e6 * MODEL_ND.disturbance)
