import crocoddyl
import pinocchio
import example_robot_data

from test_utils import NUMDIFF_MODIFIER, assertNumDiff


# Create robot model and data
ROBOT_MODEL = example_robot_data.loadICub().model
ROBOT_DATA = ROBOT_MODEL.createData()

# Create differential action model and data; link the contact data
ROBOT_STATE = crocoddyl.StateMultibody(ROBOT_MODEL)
ACTUATION = crocoddyl.ActuationModelFloatingBase(ROBOT_STATE)
IMPULSES = crocoddyl.ImpulseModelMultiple(ROBOT_STATE)
IMPULSE_6D = crocoddyl.ImpulseModel6D(ROBOT_STATE, ROBOT_MODEL.getFrameId('r_sole'))
IMPULSE_3D = crocoddyl.ImpulseModel3D(ROBOT_STATE, ROBOT_MODEL.getFrameId('l_sole'))
IMPULSES.addImpulse("r_sole_impulse", IMPULSE_6D)
IMPULSES.addImpulse("l_sole_impulse", IMPULSE_3D)
COSTS = crocoddyl.CostModelSum(ROBOT_STATE, 0)

COSTS.addCost("impulse_com", crocoddyl.CostModelImpulseCoM(ROBOT_STATE), 1.)
MODEL = crocoddyl.ActionModelImpulseFwdDynamics(ROBOT_STATE, IMPULSES, COSTS, 0., 0., True)
DATA = MODEL.createData()

# Created DAM numdiff
MODEL_ND = crocoddyl.ActionModelNumDiff(MODEL)
MODEL_ND.disturbance *= 10
dnum = MODEL_ND.createData()

x = ROBOT_STATE.rand()
u = pinocchio.utils.rand(0)  # (ACTUATION.nu)
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
