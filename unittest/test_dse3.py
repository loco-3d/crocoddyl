import pinocchio
import rospkg
import numpy as np

rospack = rospkg.RosPack()
MODEL_PATH = rospack.get_path('talos_data')
MESH_DIR = MODEL_PATH
URDF_FILENAME = "talos_left_arm.urdf"
URDF_MODEL_PATH = MODEL_PATH + "/robots/" + URDF_FILENAME


robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(URDF_MODEL_PATH, [MESH_DIR])

rmodel = robot.model
rdata = rmodel.createData()
jid = rmodel.getJointId('gripper_left_joint')

q0 = pinocchio.randomConfiguration(rmodel)

Mref = pinocchio.SE3.Random()

def df_dq(model,func,q,h=1e-8):
    dq = np.zeros((model.nv,1))
    f0 = func(q)
    res = np.zeros([len(f0),model.nv])
    for iq in range(model.nv):
        dq[iq] = h
        res[:,iq] = np.array((func(pinocchio.integrate(model,q,dq)) - f0)/h).squeeze()
        dq[iq] = 0
    return res

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

d1 = dresidualLocal(q0)
d2 = dresidualWorld(q0)
d3 = df_dq(rmodel, residualrMi, q0)

pinocchio.forwardKinematics(rmodel,rdata,q0)
oMi = rdata.oMi[jid]

assert(np.isclose(d1,d3,atol=1e-8).all())
#assert(np.isclose(d2, oMi.action.dot(d3), atol=1e-8).all())
