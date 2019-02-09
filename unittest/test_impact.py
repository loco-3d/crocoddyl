from crocoddyl import loadTalosArm,loadTalosLegs
from crocoddyl import ActionModelImpact,ImpulseModel6D,ImpulseModelMultiple
from crocoddyl import ActionModelNumDiff
from crocoddyl import m2a, a2m, absmax, absmin
from numpy.linalg import norm

# --- TALOS ARM
robot = loadTalosArm(freeFloating=False)
rmodel = robot.model

opPointName = 'root_joint'
contactName = 'arm_left_7_joint'
contactName = 'gripper_left_fingertip_1_link'

#opPointName,contactName = contactName,opPointName
CONTACTFRAME = rmodel.getFrameId(contactName)
OPPOINTFRAME = rmodel.getFrameId(opPointName)

impulseModel = ImpulseModel6D(rmodel,rmodel.getFrameId(contactName))
model  = ActionModelImpact(rmodel,impulseModel)
model.impulseWeight = 1.
data = model.createData()

x = model.State.rand()
#x[7:13] = 0
q = a2m(x[:model.nq])
v = a2m(x[model.nq:])


model.calc(data,x)
model.calcDiff(data,x)

mnum = ActionModelNumDiff(model,withGaussApprox=True)
dnum = mnum.createData()

nx,ndx,nq,nv,nu = model.nx,model.ndx,model.nq,model.nv,model.nu

mnum.calcDiff(dnum,x,None)
assert( absmax(dnum.Fx[:nv,:nv]-data.Fx[:nv,:nv]) < 1e-3 )  # dq/dq
assert( absmax(dnum.Fx[:nv,nv:]-data.Fx[:nv,nv:]) < 1e-3 )  # dq/dv
assert( absmax(dnum.Fx[nv:,nv:]-data.Fx[nv:,nv:]) < 1e-3 )  # dv/dv

import pinocchio
from pinocchio.utils import *
rdata = rmodel.createData()
#              = -K^-1 [ M'(vnext-v) + J'^T f ]
#                      [ J' vnext             ]

# Check M'(vnext-v)
def df_dq(model,func,q,h=1e-9):
    dq = zero(model.nv)
    f0 = func(q)
    res = zero([len(f0),model.nv])
    for iq in range(model.nv):
        dq[iq] = h
        res[:,iq] = (func(pinocchio.integrate(model,q,dq)) - f0)/h
        dq[iq] = 0
    return res

### Check M(vnext-v)
def Mdv(q,vnext,v):
    M = pinocchio.crba(rmodel,rdata,q)
    return M*(vnext-v)

dn = df_dq(rmodel,lambda _q:Mdv(_q,a2m(data.vnext),v),q)
g = rmodel.gravity
rmodel.gravity = pinocchio.Motion.Zero()
pinocchio.computeRNEADerivatives(rmodel,rdata,q,zero(rmodel.nv),a2m(data.vnext)-v)
d = rdata.dtau_dq.copy()
rmodel.gravity = g
assert(norm(d-dn)<1e-5)

### Check J.T f
np.set_printoptions(precision=4, linewidth=200, suppress=True)

def Jtf(q,f):
    pinocchio.computeJointJacobians(rmodel,rdata,q)
    pinocchio.forwardKinematics(rmodel,rdata,q)
    pinocchio.updateFramePlacements(rmodel,rdata)
    J = pinocchio.getFrameJacobian(rmodel,rdata,CONTACTFRAME,pinocchio.ReferenceFrame.LOCAL)
    return J.T*f

dn = df_dq(rmodel,lambda _q:Jtf(_q,a2m(data.f)),q)
g = rmodel.gravity
rmodel.gravity = pinocchio.Motion.Zero()
pinocchio.computeRNEADerivatives(rmodel,rdata,q,zero(rmodel.nv),zero(rmodel.nv),data.impulse.forces)
d = rdata.dtau_dq.copy()
rmodel.gravity = g
assert(norm(d-dn)<1e-5)

### Check J.T f + M(vnext-v)
np.set_printoptions(precision=4, linewidth=200, suppress=True)

def MvJtf(q,vnext,v,f):
    M = pinocchio.crba(rmodel,rdata,q)
    pinocchio.computeJointJacobians(rmodel,rdata,q)
    pinocchio.forwardKinematics(rmodel,rdata,q)
    pinocchio.updateFramePlacements(rmodel,rdata)
    J = pinocchio.getFrameJacobian(rmodel,rdata,CONTACTFRAME,pinocchio.ReferenceFrame.LOCAL)
    return M*(vnext-v)+J.T*f

dn = df_dq(rmodel,lambda _q:MvJtf(_q,a2m(data.vnext),v,a2m(data.f)),q)
g = rmodel.gravity
rmodel.gravity = pinocchio.Motion.Zero()
pinocchio.computeRNEADerivatives(rmodel,rdata,q,zero(rmodel.nv),a2m(data.vnext)-v,data.impulse.forces)
d = rdata.dtau_dq.copy()
rmodel.gravity = g
assert(norm(d-dn)<1e-5)

### Check J vnext
def Jv(q,vnext):
    pinocchio.computeJointJacobians(rmodel,rdata,q)
    pinocchio.forwardKinematics(rmodel,rdata,q)
    pinocchio.updateFramePlacements(rmodel,rdata)
    J = pinocchio.getFrameJacobian(rmodel,rdata,CONTACTFRAME,pinocchio.ReferenceFrame.LOCAL)
    return J*vnext

dn = df_dq(rmodel,lambda _q:Jv(_q,a2m(data.vnext)),q)

assert( absmax(dnum.Fx[nv:,:nv]-data.Fx[nv:,:nv]) < 1e-3 )  # dv/dq

assert( absmax(dnum.Fx-data.Fx) < 1e-3 )
assert( absmax(dnum.Rx-data.Rx) < 1e-3 )
assert( absmax(dnum.Lx-data.Lx) < 1e-3 )
assert( data.Fu.shape[1]==0 and data.Lu.shape == (0,))

# --- TALOS LEGS
robot = loadTalosLegs()
rmodel = robot.model
rmodel.armature[6:] = 0.1

contactName = 'left_sole_link'
contactName = 'leg_right_6_joint'

CONTACTFRAME = rmodel.getFrameId(contactName)

impulse6     = ImpulseModel6D(rmodel,rmodel.getFrameId(contactName))
impulseModel = ImpulseModelMultiple(rmodel,{ "6d": impulse6 })
model        = ActionModelImpact(rmodel,impulse6)
data         = model.createData()
model.impulseWeight = 1.

x = model.State.rand()
q = a2m(x[:model.nq])
v = a2m(x[model.nq:])

model.calc(data,x)
model.calcDiff(data,x)

rdata = rmodel.createData()

### Check M(vnext-v)
def Mdv(q,vnext,v):
    M = pinocchio.crba(rmodel,rdata,q)
    return M*(vnext-v)

dn = df_dq(rmodel,lambda _q:Mdv(_q,a2m(data.vnext),v),q)
g = rmodel.gravity
rmodel.gravity = pinocchio.Motion.Zero()
pinocchio.computeRNEADerivatives(rmodel,rdata,q,zero(rmodel.nv),a2m(data.vnext)-v)
d = rdata.dtau_dq.copy()
rmodel.gravity = g
assert(norm(d-dn)<1e-4)

### Check J.T f
np.set_printoptions(precision=4, linewidth=200, suppress=True)

def Jtf(q,f):
    pinocchio.computeJointJacobians(rmodel,rdata,q)
    pinocchio.forwardKinematics(rmodel,rdata,q)
    pinocchio.updateFramePlacements(rmodel,rdata)
    J = pinocchio.getFrameJacobian(rmodel,rdata,CONTACTFRAME,pinocchio.ReferenceFrame.LOCAL)
    return J.T*f

dn = df_dq(rmodel,lambda _q:Jtf(_q,a2m(data.f)),q)
g = rmodel.gravity
rmodel.gravity = pinocchio.Motion.Zero()
pinocchio.computeRNEADerivatives(rmodel,rdata,q,zero(rmodel.nv),zero(rmodel.nv),data.impulse.forces)
d = rdata.dtau_dq.copy()
rmodel.gravity = g
assert(norm(d-dn)<1e-4)

### Check J.T f + M(vnext-v)
np.set_printoptions(precision=4, linewidth=200, suppress=True)

def MvJtf(q,vnext,v,f):
    M = pinocchio.crba(rmodel,rdata,q)
    pinocchio.computeJointJacobians(rmodel,rdata,q)
    pinocchio.forwardKinematics(rmodel,rdata,q)
    pinocchio.updateFramePlacements(rmodel,rdata)
    J = pinocchio.getFrameJacobian(rmodel,rdata,CONTACTFRAME,pinocchio.ReferenceFrame.LOCAL)
    return M*(vnext-v)+J.T*f

dn = df_dq(rmodel,lambda _q:MvJtf(_q,a2m(data.vnext),v,a2m(data.f)),q)
g = rmodel.gravity
rmodel.gravity = pinocchio.Motion.Zero()
pinocchio.computeRNEADerivatives(rmodel,rdata,q,zero(rmodel.nv),a2m(data.vnext)-v,data.impulse.forces)
d = rdata.dtau_dq.copy()
rmodel.gravity = g
assert(norm(d-dn)<1e-4)


# Check K'r-k' = [ M' (vnext-v) + J'f - M'v ]
#                [ J' vnext
def Krk(q,vnext,v,f):
    M = pinocchio.crba(rmodel,rdata,q)
    pinocchio.computeJointJacobians(rmodel,rdata,q)
    pinocchio.forwardKinematics(rmodel,rdata,q)
    pinocchio.updateFramePlacements(rmodel,rdata)
    J = pinocchio.getFrameJacobian(rmodel,rdata,CONTACTFRAME,pinocchio.ReferenceFrame.LOCAL)
    
    return np.vstack([ M*(vnext-v)+J.T*f, J*vnext ])

dn = df_dq(rmodel,lambda _q:Krk(_q,a2m(data.vnext),v,a2m(data.f)),q)
d = np.vstack([data.did_dq, data.dv_dq])
assert(norm(d-dn)<1e-4)


mnum = ActionModelNumDiff(model,withGaussApprox=True)
dnum = mnum.createData()

nx,ndx,nq,nv,nu = model.nx,model.ndx,model.nq,model.nv,model.nu

mnum.calcDiff(dnum,x,None)
assert( absmax(dnum.Fx[:nv,:nv]-data.Fx[:nv,:nv]) < 1e-3 )  # dq/dq
assert( absmax(dnum.Fx[:nv,nv:]-data.Fx[:nv,nv:]) < 1e-3 )  # dq/dv
assert( absmax(dnum.Fx[nv:,:nv]-data.Fx[nv:,:nv]) < 3e-3 )  # dv/dq
assert( absmax(dnum.Fx[nv:,nv:]-data.Fx[nv:,nv:]) < 1e-3 )  # dv/dv

assert( absmax(dnum.Fx-data.Fx) < 1e-3 )
assert( absmax(dnum.Rx-data.Rx) < 1e-3 )
assert( absmax(dnum.Lx-data.Lx) < 1e-3 )
assert( data.Fu.shape[1]==0 and data.Lu.shape == (0,))
