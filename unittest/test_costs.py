import numpy as np
from numpy.linalg import norm

import pinocchio
from crocoddyl import (ActionModelNumDiff, ActivationModelInequality, ActivationModelWeightedQuad, CostModelCoM,
                       CostModelControl, CostModelFramePlacement, CostModelFrameRotation, CostModelFrameTranslation,
                       CostModelFrameVelocity, CostModelFrameVelocityLinear, CostModelNumDiff, CostModelState,
                       CostModelSum, DifferentialActionModelFullyActuated, DifferentialActionModelNumDiff,
                       IntegratedActionModelEuler, ShootingProblem, SolverDDP, SolverKKT, StatePinocchio, a2m,
                       loadTalosArm, m2a)
from pinocchio.utils import rand, zero
from testutils import NUMDIFF_MODIFIER, assertNumDiff

robot = loadTalosArm()
qmin = robot.model.lowerPositionLimit
qmin[:7] = -1
robot.model.lowerPositionLimit = qmin
qmax = robot.model.upperPositionLimit
qmax[:7] = 1
robot.model.upperPositionLimit = qmax
rmodel = robot.model
rdata = rmodel.createData()

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv))

costModel = CostModelFrameTranslation(rmodel, rmodel.getFrameId('gripper_left_fingertip_2_link'),
                                      np.array([.5, .4, .3]))

costData = costModel.createData(rdata)

pinocchio.forwardKinematics(rmodel, rdata, q, v)
pinocchio.computeJointJacobians(rmodel, rdata, q)
pinocchio.updateFramePlacements(rmodel, rdata)

costModel.calcDiff(costData, x, u)

costModelND = CostModelNumDiff(
    costModel,
    StatePinocchio(rmodel),
    withGaussApprox=True,
    reevals=[
        lambda m, d, x, u: pinocchio.forwardKinematics(m, d, a2m(
            x[:rmodel.nq]), a2m(x[rmodel.nq:])), lambda m, d, x, u: pinocchio.computeJointJacobians(
                m, d, a2m(x[:rmodel.nq])), lambda m, d, x, u: pinocchio.updateFramePlacements(m, d)
    ])
costDataND = costModelND.createData(rdata)

costModelND.calcDiff(costDataND, x, u)

assertNumDiff(costData.Lx, costDataND.Lx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lu, costDataND.Lu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxx, costDataND.Lxx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxu, costDataND.Lxu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Luu, costDataND.Luu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv))

costModel = CostModelFrameRotation(rmodel, rmodel.getFrameId('gripper_left_fingertip_2_link'), np.eye(3))

costData = costModel.createData(rdata)

pinocchio.forwardKinematics(rmodel, rdata, q, v)
pinocchio.computeJointJacobians(rmodel, rdata, q)
pinocchio.updateFramePlacements(rmodel, rdata)

costModel.calcDiff(costData, x, u)

costModelND = CostModelNumDiff(
    costModel,
    StatePinocchio(rmodel),
    withGaussApprox=True,
    reevals=[
        lambda m, d, x, u: pinocchio.forwardKinematics(m, d, a2m(
            x[:rmodel.nq]), a2m(x[rmodel.nq:])), lambda m, d, x, u: pinocchio.computeJointJacobians(
                m, d, a2m(x[:rmodel.nq])), lambda m, d, x, u: pinocchio.updateFramePlacements(m, d)
    ])
costDataND = costModelND.createData(rdata)

costModelND.calcDiff(costDataND, x, u)

assertNumDiff(costData.Lx, costDataND.Lx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lu, costDataND.Lu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxx, costDataND.Lxx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxu, costDataND.Lxu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Luu, costDataND.Luu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv))

costModel = CostModelFrameVelocity(rmodel, rmodel.getFrameId('gripper_left_fingertip_2_link'))
costData = costModel.createData(rdata)

pinocchio.forwardKinematics(rmodel, rdata, q, v)
pinocchio.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, zero(rmodel.nv))
pinocchio.updateFramePlacements(rmodel, rdata)

costModel.calcDiff(costData, x, u)

costModelND = CostModelNumDiff(
    costModel,
    StatePinocchio(rmodel),
    withGaussApprox=True,
    reevals=[
        lambda m, d, x, u: pinocchio.forwardKinematics(m, d, a2m(x[:rmodel.nq]), a2m(x[rmodel.nq:])), lambda m, d, x,
        u: pinocchio.computeForwardKinematicsDerivatives(m, d, a2m(x[:rmodel.nq]), a2m(x[rmodel.nq:]), zero(rmodel.nv)
                                                         ), lambda m, d, x, u: pinocchio.updateFramePlacements(m, d)
    ])
costDataND = costModelND.createData(rdata)

costModelND.calcDiff(costDataND, x, u)

assertNumDiff(costData.Lx, costDataND.Lx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 2.11e-6, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lu, costDataND.Lu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 2.11e-6, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxx, costDataND.Lxx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 2.11e-6, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxu, costDataND.Lxu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 2.11e-6, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Luu, costDataND.Luu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 2.11e-6, is now 2.11e-4 (see assertNumDiff.__doc__)

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv))

costModel = CostModelFrameVelocityLinear(rmodel, rmodel.getFrameId('gripper_left_fingertip_2_link'))
costData = costModel.createData(rdata)

pinocchio.forwardKinematics(rmodel, rdata, q, v)
pinocchio.computeForwardKinematicsDerivatives(rmodel, rdata, q, v, zero(rmodel.nv))
pinocchio.updateFramePlacements(rmodel, rdata)

costModel.calcDiff(costData, x, u)

costModelND = CostModelNumDiff(
    costModel,
    StatePinocchio(rmodel),
    withGaussApprox=True,
    reevals=[
        lambda m, d, x, u: pinocchio.forwardKinematics(m, d, a2m(x[:rmodel.nq]), a2m(x[rmodel.nq:])), lambda m, d, x,
        u: pinocchio.computeForwardKinematicsDerivatives(m, d, a2m(x[:rmodel.nq]), a2m(x[rmodel.nq:]), zero(rmodel.nv)
                                                         ), lambda m, d, x, u: pinocchio.updateFramePlacements(m, d)
    ])
costDataND = costModelND.createData(rdata)

costModelND.calcDiff(costDataND, x, u)

assertNumDiff(costData.Lx, costDataND.Lx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lu, costDataND.Lu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxx, costDataND.Lxx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxu, costDataND.Lxu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Luu, costDataND.Luu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv))

costModel = CostModelFramePlacement(rmodel, rmodel.getFrameId('gripper_left_fingertip_2_link'),
                                    pinocchio.SE3(pinocchio.SE3.Random().rotation,
                                                  np.matrix([.5, .4, .3]).T))

costData = costModel.createData(rdata)

pinocchio.forwardKinematics(rmodel, rdata, q, v)
pinocchio.computeJointJacobians(rmodel, rdata, q)
pinocchio.updateFramePlacements(rmodel, rdata)

costModel.calcDiff(costData, x, u)

costModelND = CostModelNumDiff(
    costModel,
    StatePinocchio(rmodel),
    withGaussApprox=True,
    reevals=[
        lambda m, d, x, u: pinocchio.forwardKinematics(m, d, a2m(
            x[:rmodel.nq]), a2m(x[rmodel.nq:])), lambda m, d, x, u: pinocchio.computeJointJacobians(
                m, d, a2m(x[:rmodel.nq])), lambda m, d, x, u: pinocchio.updateFramePlacements(m, d)
    ])
costDataND = costModelND.createData(rdata)

costModelND.calcDiff(costDataND, x, u)

assertNumDiff(costData.Lx, costDataND.Lx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lu, costDataND.Lu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxx, costDataND.Lxx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxu, costDataND.Lxu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Luu, costDataND.Luu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv))

costModel = CostModelCoM(rmodel, np.array([.5, .4, .3]))

costData = costModel.createData(rdata)

pinocchio.jacobianCenterOfMass(rmodel, rdata, q, False)

costModel.calcDiff(costData, x, u)

costModelND = CostModelNumDiff(
    costModel,
    StatePinocchio(rmodel),
    withGaussApprox=True,
    reevals=[lambda m, d, x, u: pinocchio.jacobianCenterOfMass(m, d, a2m(x[:rmodel.nq]), False)])
costDataND = costModelND.createData(rdata)

costModelND.calcDiff(costDataND, x, u)

assertNumDiff(costData.Lx, costDataND.Lx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lu, costDataND.Lu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxx, costDataND.Lxx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxu, costDataND.Lxu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Luu, costDataND.Luu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

X = StatePinocchio(rmodel)
q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv))
act = ActivationModelWeightedQuad(weights=np.array([2] * rmodel.nv + [.5] * rmodel.nv))
costModel = CostModelState(rmodel, X, X.rand(), activation=act)
costData = costModel.createData(rdata)
costModel.calcDiff(costData, x, u)

costModelND = CostModelNumDiff(costModel, X, withGaussApprox=True, reevals=[])
costDataND = costModelND.createData(rdata)
costModelND.calcDiff(costDataND, x, u)

assertNumDiff(costData.Lx, costDataND.Lx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lu, costDataND.Lu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxx, costDataND.Lxx * costModel.activation.weights, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxu, costDataND.Lxu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Luu, costDataND.Luu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

X = StatePinocchio(rmodel)
q = a2m(np.random.rand(rmodel.nq))  # random value between 0 and 1
u = m2a(np.random.rand(rmodel.nv))  # random value between 0 and 1
v = a2m(np.random.rand(rmodel.nv))

x = m2a(np.concatenate([q, v]))

lowerLimit = np.array([
    0.3,
] * (rmodel.nq + rmodel.nv))
upperLimit = np.array([
    0.7,
] * (rmodel.nq + rmodel.nv))
act_ineq = ActivationModelInequality(lowerLimit=lowerLimit, upperLimit=upperLimit, beta=1.0)
costModel = CostModelState(rmodel, X, ref=X.zero(), activation=act_ineq)

costData = costModel.createData(rdata)
costModel.calc(costData, x, u)

costModel.calcDiff(costData, x, u)

costModelND = CostModelNumDiff(costModel, X, withGaussApprox=True, reevals=[])
costDataND = costModelND.createData(rdata)
costModelND.calcDiff(costDataND, x, u)

assertNumDiff(costData.Lx, costDataND.Lx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lu, costDataND.Lu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
# Check that the cost derivative is zero if q>=lower and q<=upper
# and that cost is positive if q<lower or q>upper
lowersafe = m2a(x) >= lowerLimit
uppersafe = m2a(x) <= upperLimit

assert ((costData.Lx[lowersafe & uppersafe] == 0.).all())
assert ((costData.Lx[~lowersafe & ~uppersafe] != 0.).all())

assert ((costData.Lxx.diagonal()[~lowersafe] == 1.).all())
assert ((costData.Lxx.diagonal()[~uppersafe] == 1.).all())
assert ((costData.Lxx.diagonal()[uppersafe & lowersafe] == 0.).all())
# --------------------------Check Inf joint limits

lowerLimit[:rmodel.nq] = -np.inf  # inf position lower limit
upperLimit[-rmodel.nv:] = np.inf  # inf velocity upper limit
act_ineq = ActivationModelInequality(lowerLimit=lowerLimit, upperLimit=upperLimit, beta=None)

costModel = CostModelState(rmodel, X, ref=X.zero(), activation=act_ineq)

costData = costModel.createData(rdata)
costModel.calc(costData, x, u)

costModel.calcDiff(costData, x, u)

costModelND = CostModelNumDiff(costModel, X, withGaussApprox=True, reevals=[])
costDataND = costModelND.createData(rdata)
costModelND.calcDiff(costDataND, x, u)

assertNumDiff(costData.Lx, costDataND.Lx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lu, costDataND.Lu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
# Check that the cost derivative is zero if q>=lower and q<=upper
# and that cost is positive if q<lower or q>upper
lowersafe = m2a(x) >= lowerLimit
uppersafe = m2a(x) <= upperLimit

assert ((costData.Lx[lowersafe & uppersafe] == 0.).all())
assert ((costData.Lx[~lowersafe & ~uppersafe] != 0.).all())

X = StatePinocchio(rmodel)
q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv))

costModel = CostModelControl(rmodel)
costData = costModel.createData(rdata)
costModel.calcDiff(costData, x, u)

costModelND = CostModelNumDiff(costModel, StatePinocchio(rmodel), withGaussApprox=True)
costDataND = costModelND.createData(rdata)
costModelND.calcDiff(costDataND, x, u)

assertNumDiff(costData.Lx, costDataND.Lx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lu, costDataND.Lu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxx, costDataND.Lxx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxu, costDataND.Lxu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Luu, costDataND.Luu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

X = StatePinocchio(rmodel)
q = pinocchio.randomConfiguration(rmodel)
v = rand(rmodel.nv)
x = m2a(np.concatenate([q, v]))
u = m2a(rand(rmodel.nv))
act1 = ActivationModelWeightedQuad(weights=np.array([
    1.,
] * x.size))
cost1 = CostModelFrameTranslation(rmodel, rmodel.getFrameId('gripper_left_fingertip_2_link'), np.array([.5, .4, .3]))
cost2 = CostModelState(rmodel, X, X.rand(), activation=act1)
cost3 = CostModelControl(rmodel)

costModel = CostModelSum(rmodel)
costModel.addCost("pos", cost1, 10)
costModel.addCost("regx", cost2, .1)
costModel.addCost("regu", cost3, .01)
costData = costModel.createData(rdata)

pinocchio.forwardKinematics(rmodel, rdata, q, v)
pinocchio.computeJointJacobians(rmodel, rdata, q)
pinocchio.updateFramePlacements(rmodel, rdata)
costModel.calcDiff(costData, x, u)

costModelND = CostModelNumDiff(
    costModel,
    StatePinocchio(rmodel),
    withGaussApprox=True,
    reevals=[
        lambda m, d, x, u: pinocchio.forwardKinematics(m, d, a2m(
            x[:rmodel.nq]), a2m(x[rmodel.nq:])), lambda m, d, x, u: pinocchio.computeJointJacobians(
                m, d, a2m(x[:rmodel.nq])), lambda m, d, x, u: pinocchio.updateFramePlacements(m, d)
    ])
costDataND = costModelND.createData(rdata)
costModelND.calcDiff(costDataND, x, u)

assertNumDiff(costData.Lx, costDataND.Lx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lu, costDataND.Lu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxx, costDataND.Lxx, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Lxu, costDataND.Lxu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(costData.Luu, costDataND.Luu, NUMDIFF_MODIFIER *
              costModelND.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)


class DifferentialActionModelPositioning(DifferentialActionModelFullyActuated):
    def __init__(self, pinocchioModel, frameName='gripper_left_fingertip_2_link'):
        costModel = CostModelSum(pinocchioModel)
        DifferentialActionModelFullyActuated.__init__(self, pinocchioModel, costModel)
        self.costs.addCost(name="pos",
                           weight=10,
                           cost=CostModelFrameTranslation(pinocchioModel, pinocchioModel.getFrameId(frameName),
                                                          np.array([.5, .4, .3])))
        self.costs.addCost(name="regx",
                           weight=0.1,
                           cost=CostModelState(pinocchioModel, self.State, self.State.zero(), activation=act1))
        self.costs.addCost(name="regu", weight=0.01, cost=CostModelControl(pinocchioModel))


q = m2a(pinocchio.randomConfiguration(rmodel))
v = np.random.rand(rmodel.nv) * 2 - 1
x = np.concatenate([q, v])
u = np.random.rand(rmodel.nv) * 2 - 1

model = DifferentialActionModelPositioning(rmodel)
data = model.createData()

a, _ = model.calc(data, x, u)
model.calcDiff(data, x, u)

mnum = DifferentialActionModelNumDiff(model, withGaussApprox=True)
dnum = mnum.createData()

model.calcDiff(data, x, u)
mnum.calcDiff(dnum, x, u)
assertNumDiff(data.Fx, dnum.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Rx, dnum.Rx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Ru, dnum.Ru,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)

dmodel = DifferentialActionModelPositioning(rmodel)
ddata = dmodel.createData()
model = IntegratedActionModelEuler(dmodel)
data = model.createData()

x = model.State.zero()
u = np.zeros(model.nu)
xn, c = model.calc(data, x, u)

model.timeStep = 1
model.differential.costs
for k in model.differential.costs.costs.keys():
    model.differential.costs[k].weight = 1

model.calcDiff(data, x, u)

mnum = ActionModelNumDiff(model, withGaussApprox=True)
dnum = mnum.createData()

mnum.calcDiff(dnum, x, u)
assertNumDiff(data.Fx, dnum.Fx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 3.16e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Fu, dnum.Fu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 3.16e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lx, dnum.Lx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 3.16e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(data.Lu, dnum.Lu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 3.16e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Lxx, data.Lxx,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 3.16e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Lxu, data.Lxu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 3.16e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dnum.Luu, data.Luu,
              NUMDIFF_MODIFIER * mnum.disturbance)  # threshold was 3.16e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
# --- DDP FOR THE ARM ---
dmodel = DifferentialActionModelPositioning(rmodel)
model = IntegratedActionModelEuler(dmodel)

problem = ShootingProblem(model.State.zero() + 1, [model], model)
kkt = SolverKKT(problem)
kkt.th_stop = 1e-18
xkkt, ukkt, dkkt = kkt.solve()

ddp = SolverDDP(problem)
ddp.th_stop = 1e-18
xddp, uddp, dddp = ddp.solve()

assert (norm(uddp[0] - ukkt[0]) < 1e-6)
