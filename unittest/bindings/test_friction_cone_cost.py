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

# Create friction cone and its activation
frictionCone = crocoddyl.FrictionCone(np.matrix([0., 0., 1.]).T, 0.7, 4, False)
activation = crocoddyl.ActivationModelQuadraticBarrier(crocoddyl.ActivationBounds(frictionCone.lb, frictionCone.ub))

# Contact friction-cone cost unittest
CONTACTS = crocoddyl.ContactModelMultiple(ROBOT_STATE, ACTUATION.nu)
CONTACT_6D = crocoddyl.ContactModel6D(
    ROBOT_STATE, crocoddyl.FramePlacement(ROBOT_MODEL.getFrameId('r_sole'), pinocchio.SE3.Random()), ACTUATION.nu,
    pinocchio.utils.rand(2))
CONTACT_3D = crocoddyl.ContactModel3D(
    ROBOT_STATE, crocoddyl.FrameTranslation(ROBOT_MODEL.getFrameId('l_sole'),
                                            pinocchio.SE3.Random().translation), ACTUATION.nu, pinocchio.utils.rand(2))
CONTACTS.addContact("r_sole_contact", CONTACT_6D)
CONTACTS.addContact("l_sole_contact", CONTACT_3D)
COSTS = crocoddyl.CostModelSum(ROBOT_STATE, ACTUATION.nu)
COSTS.addCost(
    "r_sole_friction_cone",
    crocoddyl.CostModelContactFrictionCone(ROBOT_STATE, activation,
                                           crocoddyl.FrameFrictionCone(ROBOT_MODEL.getFrameId('r_sole'), frictionCone),
                                           ACTUATION.nu), 1.)
COSTS.addCost(
    "l_sole_friction_cone",
    crocoddyl.CostModelContactFrictionCone(ROBOT_STATE, activation,
                                           crocoddyl.FrameFrictionCone(ROBOT_MODEL.getFrameId('l_sole'), frictionCone),
                                           ACTUATION.nu), 1.)
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

# Impulse friction-cone cost unittest
IMPULSES = crocoddyl.ImpulseModelMultiple(ROBOT_STATE)
IMPULSE_6D = crocoddyl.ImpulseModel6D(ROBOT_STATE, ROBOT_MODEL.getFrameId('r_sole'))
IMPULSE_3D = crocoddyl.ImpulseModel3D(ROBOT_STATE, ROBOT_MODEL.getFrameId('l_sole'))
IMPULSES.addImpulse("r_sole_impulse", IMPULSE_6D)
IMPULSES.addImpulse("l_sole_impulse", IMPULSE_3D)
COSTS = crocoddyl.CostModelSum(ROBOT_STATE, 0)
COSTS.addCost(
    "r_sole_friction_cone",
    crocoddyl.CostModelImpulseFrictionCone(ROBOT_STATE, activation,
                                           crocoddyl.FrameFrictionCone(ROBOT_MODEL.getFrameId('r_sole'),
                                                                       frictionCone)), 1.)
COSTS.addCost(
    "l_sole_friction_cone",
    crocoddyl.CostModelImpulseFrictionCone(ROBOT_STATE, activation,
                                           crocoddyl.FrameFrictionCone(ROBOT_MODEL.getFrameId('l_sole'),
                                                                       frictionCone)), 1.)
MODEL = crocoddyl.ActionModelImpulseFwdDynamics(ROBOT_STATE, IMPULSES, COSTS, 0., 0., True)
DATA = MODEL.createData()

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
