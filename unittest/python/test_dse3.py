import numpy as np

import eigenpy
import pinocchio
from crocoddyl import loadTalosArm
from crocoddyl.utils import EPS
from testutils import NUMDIFF_MODIFIER, assertNumDiff, df_dq

eigenpy.switchToNumpyMatrix()

robot = loadTalosArm()
rmodel = robot.model
rdata = rmodel.createData()
jid = rmodel.getJointId('gripper_left_joint')

q0 = pinocchio.randomConfiguration(rmodel)

Mref = pinocchio.SE3.Random()


def residualrMi(q):
    pinocchio.forwardKinematics(rmodel, rdata, q)
    return pinocchio.log(Mref.inverse() * rdata.oMi[jid]).vector


def dresidualLocal(q):
    pinocchio.forwardKinematics(rmodel, rdata, q)
    pinocchio.computeJointJacobians(rmodel, rdata, q)
    rMi = Mref.inverse() * rdata.oMi[jid]
    return np.dot(pinocchio.Jlog6(rMi), pinocchio.getJointJacobian(rmodel, rdata, jid, pinocchio.ReferenceFrame.LOCAL))


def dresidualWorld(q):
    pinocchio.forwardKinematics(rmodel, rdata, q)
    pinocchio.computeJointJacobians(rmodel, rdata, q)
    rMi = Mref.inverse() * rdata.oMi[jid]
    return np.dot(pinocchio.Jlog6(rMi), pinocchio.getJointJacobian(rmodel, rdata, jid, pinocchio.ReferenceFrame.WORLD))


d1 = dresidualLocal(q0)
d2 = dresidualWorld(q0)
d3 = df_dq(rmodel, residualrMi, q0)

pinocchio.forwardKinematics(rmodel, rdata, q0)
oMi = rdata.oMi[jid]

h = np.sqrt(2 * EPS)
assertNumDiff(d1, d3, NUMDIFF_MODIFIER * h)  # threshold was 1e-4, is now 2.11e-4 (see assertNumDiff.__doc__)
