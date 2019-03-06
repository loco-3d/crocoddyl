import pinocchio
import numpy as np
from testutils import df_dq

from crocoddyl import loadTalosArm
robot = loadTalosArm()
rmodel = robot.model
rdata = rmodel.createData()
jid = rmodel.getJointId('gripper_left_joint')

q0 = pinocchio.randomConfiguration(rmodel)

Mref = pinocchio.SE3.Random()

def residualrMi(q):
  pinocchio.forwardKinematics(rmodel,rdata,q)
  return pinocchio.log(Mref.inverse()*rdata.oMi[jid]).vector

def dresidualLocal(q):
  pinocchio.forwardKinematics(rmodel,rdata,q)
  pinocchio.computeJointJacobians(rmodel,rdata,q)
  rMi = Mref.inverse()*rdata.oMi[jid]
  return np.dot(pinocchio.Jlog6(rMi),
                   pinocchio.getJointJacobian(rmodel, rdata, jid,
                                              pinocchio.ReferenceFrame.LOCAL))

def dresidualWorld(q):
  pinocchio.forwardKinematics(rmodel,rdata,q)
  pinocchio.computeJointJacobians(rmodel,rdata,q)
  rMi = Mref.inverse()*rdata.oMi[jid]
  return np.dot(pinocchio.Jlog6(rMi),
                   pinocchio.getJointJacobian(rmodel, rdata, jid,
                                           pinocchio.ReferenceFrame.WORLD))

eps = 1e-8
d1 = dresidualLocal(q0)
d2 = dresidualWorld(q0)
d3 = df_dq(rmodel, residualrMi, q0, h=eps)

pinocchio.forwardKinematics(rmodel,rdata,q0)
oMi = rdata.oMi[jid]

assert(np.isclose(d1,d3,atol=np.sqrt(eps)).all())
#assert(np.isclose(d2, oMi.action.dot(d3), atol=1e-8).all())
