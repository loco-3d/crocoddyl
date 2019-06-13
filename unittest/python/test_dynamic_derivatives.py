from functools import partial

import numpy as np
import pinocchio
from crocoddyl import loadTalosArm
from crocoddyl.utils import EPS
from numpy.linalg import inv, norm, pinv
from pinocchio import aba, rnea
from pinocchio.utils import cross, rand, skew, zero
from testutils import NUMDIFF_MODIFIER, assertNumDiff, df_dq, df_dx

robot = loadTalosArm()
model = robot.model
data = model.createData()


def absmax(A):
    return np.max(abs(A))


def absmin(A):
    return np.min(abs(A))


# if np.any(np.isinf(model.upperPositionLimit)):
# qmin = rmodel.lowerPositionLimit
# qmin[:7] = -1
# rmodel.lowerPositionLimit = qmin
# qmax = rmodel.upperPositionLimit
# qmax[:7] = 1
# rmodel.upperPositionLimit = qmax

q = pinocchio.randomConfiguration(model)
v = rand(model.nv) * 2 - 1
a = rand(model.nv) * 2 - 1
tau = pinocchio.rnea(model, data, q, v, a)

# Basic check of calling the derivatives.
pinocchio.computeABADerivatives(model, data, q, v, tau)
pinocchio.computeRNEADerivatives(model, data, q, v, a)
'''
a = M^-1 ( tau - b)
a'  =  -M^-1 M' M^-1 (tau-b) - M^-1 b'
    =  -M^-1 M' a - M^-1 b'
    =  -M^-1 (M'a + b')
    =  -M^-1 tau'
'''

# Check that a = J aq + Jdot vq

M = data.M
Mi = pinocchio.computeMinverse(model, data, q)
da = data.ddq_dq
dtau = data.dtau_dq

assert (absmax(M * da + dtau) < 1e-9)
assert (absmax(da + Mi * dtau) < 1e-9)

pinocchio.forwardKinematics(model, data, q, v, a)
pinocchio.updateFramePlacements(model, data)

frameJacobian = pinocchio.getFrameJacobian(model, data, 10, pinocchio.ReferenceFrame.LOCAL)
frameJacobianTimeVariation = pinocchio.getFrameJacobianTimeVariation(model, data, 10, pinocchio.ReferenceFrame.LOCAL)
frameAcceleration = pinocchio.getFrameAcceleration(model, data, 10)
assert (absmax(frameJacobian * a + frameJacobianTimeVariation * v - frameAcceleration.vector) < 1e-9)
'''
(a,f) = K^-1 (tau-b,-gamma)
avec K = [ M J* ; J 0 ]

(a',f') = -K^-1 K' K^-1 (tau-b,-gamma) - K^-1 (b';gamma')
        = -Ki   K' (a,f) - Ki (b';g')
        = -Ki   (  K'(a,f) - (b',g') )

'''

# Define finite-diff routines.

# Check ABA derivatives (without forces)

da_dq = df_dq(model, lambda q_: pinocchio.aba(model, data, q_, v, tau), q)
da_dv = df_dx(lambda v_: pinocchio.aba(model, data, q, v_, tau), v)
pinocchio.computeABADerivatives(model, data, q, v, tau)

h = np.sqrt(2 * EPS)
assertNumDiff(da_dq, data.ddq_dq,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-2, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(da_dv, data.ddq_dv,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-2, is now 2.11e-4 (see assertNumDiff.__doc__)

# Check RNEA Derivatives (without forces)

a = pinocchio.aba(model, data, q, v, tau)
dtau_dq = df_dq(model, lambda q_: pinocchio.rnea(model, data, q_, v, a), q)
pinocchio.computeRNEADerivatives(model, data, q, v, a)

assertNumDiff(dtau_dq, data.dtau_dq,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

# Check RNEA versus ABA derivatives.

Mi = pinocchio.computeMinverse(model, data, q)
assert (absmax(Mi - inv(data.M)) < 1e-6)

D = np.dot(Mi, data.dtau_dq)
assertNumDiff(D, -data.ddq_dq, NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

# ---- ABA AND RNEA with forces

# Set forces container
fs = pinocchio.StdVect_Force()
for i in range(model.njoints):
    fs.append(pinocchio.Force.Random())

# Check RNEA derivatives versus finite-diff (with forces)
a = pinocchio.aba(model, data, q, v, tau, fs)
dtau_dqn = df_dq(model, lambda q_: pinocchio.rnea(model, data, q_, v, a, fs), q)
pinocchio.computeRNEADerivatives(model, data, q, v, a, fs)
dtau_dq = data.dtau_dq.copy()
assertNumDiff(dtau_dqn, dtau_dq,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

# Check ABA derivatives versus finite diff (with forces)
da_dqn = df_dq(model, lambda q_: pinocchio.aba(model, data, q_, v, tau, fs), q)
pinocchio.computeABADerivatives(model, data, q, v, tau, fs)
da_dq = data.ddq_dq.copy()
assertNumDiff(da_dq, da_dqn, NUMDIFF_MODIFIER * h)  # threshold was 3e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

# Check ABA versus RNEA derivatives (with forces)
assertNumDiff(inv(data.M) * dtau_dq, -da_dq,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

# Check ABA versus RNEA + forces (no derivatives)
del a
for i, f in enumerate(fs[:-1]):
    fs[i] = f * 0
f = fs[-1].vector
M = pinocchio.crba(model, data, q).copy()
pinocchio.computeJointJacobians(model, data, q)
J = pinocchio.getJointJacobian(model, data, model.joints[-1].id, pinocchio.ReferenceFrame.LOCAL).copy()
b = pinocchio.rnea(model, data, q, v, zero(model.nv)).copy()
a = pinocchio.aba(model, data, q, v, tau, fs).copy()
assert (absmax(a - (inv(M) * (tau - b + J.T * f))) < 1e-6)

tau = pinocchio.rnea(model, data, q, v, a, fs)
assert (absmax(tau - (M * a + b - J.T * f)) < 1e-6)

# ------------------------------------------------
# ------------------------------------------------
# ------------------------------------------------

# Checking linear acceleration (i.e. ahat = a.linear + w x v = [a + v x (vlinear,0)].linear )

a = rand(model.nv) * 2 - 1
dt = 1e-6
Motion = pinocchio.Motion
O3 = zero(3)

# --- check lin vel
pinocchio.forwardKinematics(model, data, q, v)
iv = data.v[-1].copy()
M = data.oMi[-1].copy()
r = data.oMi[-1].translation

qp = pinocchio.integrate(model, q, v * dt)
pinocchio.forwardKinematics(model, data, qp)
rp = data.oMi[-1].translation
assert (absmax(M.rotation.T * (rp - r) / dt - iv.linear) < 10 * dt)

# --- check lin acc
pinocchio.forwardKinematics(model, data, q, v, a)
iv = data.v[-1].copy()
ia = data.a[-1].copy()
M = data.oMi[-1].copy()
R = M.rotation
r = data.oMi[-1].translation
oa = M * ia
ov = R * iv.linear
ow = R * iv.angular

vp = v + a * dt
pinocchio.forwardKinematics(model, data, qp, vp)
ivp = data.v[-1].copy()
ovp = data.oMi[-1].rotation * data.v[-1].linear

rdot = ov
rddot = (ovp - ov) / dt

ia + Motion(rdot, O3).cross(iv)

assert (absmax(R * ia.linear - (rddot - cross(ow, rdot))) < 10 * dt)

# --- check lin acc
vq = rand(model.nv) * 2 - 1
aq = rand(model.nv) * 2 - 1

# alpha = [ rddot, wdot ] + [rdot,0] x nu
# alpha_l = rddot - w x rdot

pinocchio.forwardKinematics(model, data, q, vq, aq)
iv = data.v[-1].copy()
ia = data.a[-1].copy()
M = data.oMi[-1].copy()
R = M.rotation
r = data.oMi[-1].translation
v = iv.linear
w = iv.angular

vqp = vq + aq * dt
qp = pinocchio.integrate(model, q, vq * dt)
pinocchio.forwardKinematics(model, data, qp, vqp)
vp = R.T * data.oMi[-1].rotation * data.v[-1].linear

rdot = v
rddot = (vp - v) / dt

assert (absmax(ia.linear - (rddot - cross(w, rdot))) < 10 * dt)
assert (absmax(rddot - (ia.linear + cross(w, rdot))) < 10 * dt)
assert (absmax(rddot - (ia + iv.cross(Motion(rdot, O3))).linear) < 10 * dt)

# Check q derivatives of linear acceleration
q = pinocchio.randomConfiguration(model)
vq = rand(model.nv) * 2 - 1
aq = rand(model.nv) * 2 - 1

pinocchio.forwardKinematics(model, data, q, vq, aq)
M = data.oMi[-1].copy()
v = data.v[-1].copy()
a = data.a[-1].copy()

vv = v.linear.copy()
vw = v.angular.copy()
aa = a.linear + cross(vw, vv)
assert (absmax(aa - (a + v.cross(Motion(vv, zero(3)))).linear) < 1e-9)


def calcaa(q, vq, aq):
    pinocchio.forwardKinematics(model, data, q, vq, aq)
    return data.a[-1].linear + cross(data.v[-1].angular, data.v[-1].linear)


def calca(q, vq, aq):
    pinocchio.forwardKinematics(model, data, q, vq, aq)
    return data.a[-1].vector


def calcwxv(q, vq, aq):
    pinocchio.forwardKinematics(model, data, q, vq, aq)
    return cross(data.v[-1].angular, data.v[-1].linear)


daa_dqn = df_dq(model, partial(calcaa, vq=vq, aq=aq), q)
da_dqn = df_dq(model, partial(calca, vq=vq, aq=aq), q)
dwxv_dqn = df_dq(model, partial(calcwxv, vq=vq, aq=aq), q)

pinocchio.computeJointJacobians(model, data, q)
J = pinocchio.getJointJacobian(model, data, model.joints[-1].id, pinocchio.ReferenceFrame.LOCAL)
Jv = J[:3, :]
Jw = J[3:, :]
# a + wxv
pinocchio.computeForwardKinematicsDerivatives(model, data, q, vq, aq)
# da_dq = data.ddq_dq
dv_dq, da_dq, da_dv, da_da = pinocchio.getJointAccelerationDerivatives(model, data, model.joints[-1].id,
                                                                       pinocchio.ReferenceFrame.LOCAL)
assertNumDiff(da_dq, da_dqn, NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)


def calcv(q, vq, aq):
    pinocchio.forwardKinematics(model, data, q, vq, aq)
    return data.v[-1].linear


def calcw(q, vq, aq):
    pinocchio.forwardKinematics(model, data, q, vq, aq)
    return data.v[-1].angular


dv_dqn = df_dq(model, partial(calcv, vq=vq, aq=aq), q)
dw_dqn = df_dq(model, partial(calcw, vq=vq, aq=aq), q)
assertNumDiff(dv_dq[:3, :], dv_dqn,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dv_dq[3:, :], dw_dqn,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

daa_dq = da_dq[:3, :] + skew(vw) * dv_dq[:3, :] - skew(vv) * dv_dq[3:, :]
assertNumDiff(daa_dq, daa_dqn, NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

# ------------------------------------------------------------------------
# Check contact dynamics 3D contact
del vq, aq
q = pinocchio.randomConfiguration(model)
v = rand(model.nv) * 2 - 1
tau = rand(model.nv) * 2 - 1
for i, f in enumerate(fs):
    fs[i] = f * 0

pinocchio.computeJointJacobians(model, data, q)
J6 = pinocchio.getJointJacobian(model, data, model.joints[-1].id, pinocchio.ReferenceFrame.LOCAL).copy()
J = J6[:3, :]
v -= pinv(J) * J * v
assert (norm(J * v) < 1e-6)
b = pinocchio.rnea(model, data, q, v, zero(model.nv)).copy()
M = pinocchio.crba(model, data, q).copy()

# ahat = a.l + wxv
pinocchio.forwardKinematics(model, data, q, v, zero(model.nv))
gamma = data.a[-1].linear + cross(data.v[-1].angular, data.v[-1].linear)

# M a + b = tau + J'f
# J a     = -gamma

K = np.bmat([[M, J.T], [J, zero([3, 3])]])
r = np.concatenate([tau - b, -gamma])
af = inv(K) * r
a = af[:model.nv]
f = af[model.nv:]
fs[-1] = pinocchio.Force(-f, zero(3))
assert (absmax(rnea(model, data, q, v, a, fs) - (M * a + b + J.T * f)) < 1e-6)
assert (absmax(aba(model, data, q, v, tau, fs) - (inv(M) * (tau - b - J.T * f))) < 1e-6)

# Check contact-dyninv deriv
# af  = Ki r  = [ MJt; J0 ] [ tau-b;-gamma ]
# af' = -Ki K' Ki r + Ki r'
#     = -Ki ( K'  af - r' )
#     = -Ki [ M'a + Jt'f + b' ; J' a + gamma' ]

# Check M'a + J'f + b'
dtau_dqn = df_dq(model, lambda _q: rnea(model, data, _q, v, a, fs), q)
dtau_dvn = df_dq(model, lambda _v: rnea(model, data, q, _v, a, fs), v)

pinocchio.computeRNEADerivatives(model, data, q, v, a, fs)
dtau_dq = data.dtau_dq.copy()
dtau_dv = data.dtau_dv.copy()
assertNumDiff(dtau_dq, dtau_dqn,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dtau_dv, dtau_dvn,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)


# Check gamma'
def fgamma(q_, v_, a_):
    pinocchio.forwardKinematics(model, data, q_, v_, a_)
    return data.a[-1].linear + cross(data.v[-1].angular, data.v[-1].linear)


dgamma_dqn = df_dq(model, lambda _q: fgamma(_q, v, a), q)
dgamma_dvn = df_dq(model, lambda _v: fgamma(q, _v, a), v)

pinocchio.computeForwardKinematicsDerivatives(model, data, q, v, a)
dv_dq, da_dq, da_dv, da_da = pinocchio.getJointAccelerationDerivatives(model, data, model.joints[-1].id,
                                                                       pinocchio.ReferenceFrame.LOCAL)

pinocchio.computeJointJacobians(model, data, q)
J = pinocchio.getJointJacobian(model, data, model.joints[-1].id, pinocchio.ReferenceFrame.LOCAL)
vv, vw = data.v[-1].linear, data.v[-1].angular
dgamma_dq = da_dq[:3, :] + skew(vw) * dv_dq[:3, :] - skew(vv) * dv_dq[3:, :]
dgamma_dv = da_dv[:3, :] + skew(vw) * J[:3, :] - skew(vv) * J[3:, :]

assertNumDiff(dgamma_dq, dgamma_dqn,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)
assertNumDiff(dgamma_dv, dgamma_dvn,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)


# Check (Ki r)'
def cid(q_, v_, tau_):
    pinocchio.computeJointJacobians(model, data, q_)
    J6 = pinocchio.getJointJacobian(model, data, model.joints[-1].id, pinocchio.ReferenceFrame.LOCAL).copy()
    J = J6[:3, :]
    b = pinocchio.rnea(model, data, q_, v_, zero(model.nv)).copy()
    M = pinocchio.crba(model, data, q_).copy()
    pinocchio.forwardKinematics(model, data, q_, v_, zero(model.nv))
    gamma = data.a[-1].linear + cross(data.v[-1].angular, data.v[-1].linear)
    K = np.bmat([[M, J.T], [J, zero([3, 3])]])
    r = np.concatenate([tau_ - b, -gamma])
    return inv(K) * r


dcid_dqn = df_dq(model, lambda _q: cid(_q, v, tau), q)
KJn = K * dcid_dqn
KJ = -np.vstack([dtau_dq, dgamma_dq])
assertNumDiff(KJ, KJn, NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

dcid_dq = -inv(K) * np.vstack([dtau_dq, dgamma_dq])
assertNumDiff(dcid_dqn, dcid_dq,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

dcid_dvn = df_dx(lambda _v: cid(q, _v, tau), v)

dcid_dv = -inv(K) * np.vstack([dtau_dv, dgamma_dv])

assertNumDiff(dcid_dvn, dcid_dv,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

# ------------------------------------------
# Check 6d contact

q = pinocchio.randomConfiguration(model)
v = rand(model.nv) * 2 - 1
tau = rand(model.nv) * 2 - 1
for i, f in enumerate(fs):
    fs[i] = f * 0


def Kid(q_, J_=None):
    pinocchio.computeJointJacobians(model, data, q_)
    J = pinocchio.getJointJacobian(model, data, model.joints[-1].id, pinocchio.ReferenceFrame.LOCAL).copy()
    if J_ is not None:
        J_[:, :] = J
    M = pinocchio.crba(model, data, q_).copy()
    return np.bmat([[M, J.T], [J, zero([6, 6])]])


def cid2(q_, v_, tau_):
    pinocchio.computeJointJacobians(model, data, q_)
    K = Kid(q_)
    b = pinocchio.rnea(model, data, q_, v_, zero(model.nv)).copy()
    pinocchio.forwardKinematics(model, data, q_, v_, zero(model.nv))
    gamma = data.a[-1].vector
    r = np.concatenate([tau_ - b, -gamma])
    return inv(K) * r


J = zero([6, model.nv])
K = Kid(q, J)
v -= pinv(J) * J * v

af = cid2(q, v, tau)
a = af[:model.nv]
f = af[model.nv:]
fs[-1] = -pinocchio.Force(f)

pinocchio.computeRNEADerivatives(model, data, q, v, a, fs)
dtau_dq = data.dtau_dq.copy()

pinocchio.computeForwardKinematicsDerivatives(model, data, q, v, a)
dv_dq, da_dq, da_dv, da_da = pinocchio.getJointAccelerationDerivatives(model, data, model.joints[-1].id,
                                                                       pinocchio.ReferenceFrame.LOCAL)
dgamma_dq = da_dq.copy()

dcid_dqn = df_dq(model, lambda _q: cid2(_q, v, tau), q)
KJn = K * dcid_dqn
KJ = -np.vstack([dtau_dq, dgamma_dq])
assertNumDiff(KJ, KJn, NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

dcid_dq = -inv(K) * np.vstack([dtau_dq, dgamma_dq])
assertNumDiff(dcid_dqn, dcid_dq,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-3, is now 2.11e-4 (see assertNumDiff.__doc__)

dcid_dun = df_dx(lambda _u: cid2(q, v, _u), tau)
# K*D = [ I_nv; O_ncxnv ]
# D = Kinv * [ I_nv ; 0_ncxnv ] = Kinv[:nv,:]
dcid_du = inv(K)[:, :model.nv]
assertNumDiff(dcid_du, dcid_dun,
              NUMDIFF_MODIFIER * h)  # threshold was 1e-5, is now 2.11e-4 (see assertNumDiff.__doc__)
'''

# --- Jerk
# rddot  = a + v x rdot
# rdddot = adot + v x v x rdot + v x rddot
#        = adot + v x v x rdot + vx a + v x v x rdot
#        = adot + 2vxvxrdot + vxa
dt = 1e-9
jq = rand(model.nv)*2-1

pinocchio.forwardKinematics(model,data,q,vq,aq)
iv = data.v[-1]
ia = data.a[-1]
M = data.oMi[-1]
rdot  = iv.linear
rddot = (ia + iv.cross(Motion(rdot,O3))).linear
vxrd = data.v[-1].cross(Motion(data.v[-1].linear,O3)).linear

qp = pinocchio.integrate(model,q,vq*dt)
vqp = vq + aq*dt
aqp = aq + jq*dt
pinocchio.forwardKinematics(model,data,qp,vqp,aqp)
Rp = data.oMi[-1].rotation
dR = R.T*Rp
rdotp  = dR*data.v[-1].linear
rddotp = dR*data.a[-1].linear + dR*cross(data.v[-1].angular,data.v[-1].linear)
vxrdp = dR*data.v[-1].cross(Motion(data.v[-1].linear,O3)).linear

rdddot = (rddotp-rddot)/dt
vxrd_dot = (vxrdp-vxrd)/dt

d = iv.cross(iv.cross(Motion(rdot,O3))).linear + ia.cross(Motion(rdot,O3)).linear + iv.cross(Motion(rddot,O3)).linear

iv.cross



# ---- traj
assert(model.nq==model.nv)
F = rand(model.nv)*2*np.pi
P = (rand(model.nv)*2-1)*np.pi
q0 = rand(model.nv)*2-1
dq = rand(model.nv)*2-1

t = np.random.rand()
dt = 1e-6
tp = t+dt

Q = lambda t_: q0 + np.diagflat(dq)                   *np.cos(F*t_)
VQ = lambda t_:   - np.diagflat(dq)*np.diagflat(F)    *np.sin(F*t_)
AQ = lambda t_:   - np.diagflat(dq)*np.diagflat(F)**2 *np.cos(F*t_)
JQ = lambda t_:     np.diagflat(dq)*np.diagflat(F)**3 *np.sin(F*t_)

q = Q(t)
vq = VQ(t)
aq = AQ(t)
jq = JQ(t)

qp = Q(t+dt)
vqp = VQ(t+dt)
aqp = AQ(t+dt)
jqp = JQ(t+dt)


pinocchio.forwardKinematics(model,data,q,vq,aq)
iv = data.v[-1].copy()
ia = data.a[-1].copy()
M = data.oMi[-1].copy()
R = M.rotation
rdot  = iv.linear
rddot = (ia + iv.cross(Motion(rdot,O3))).linear
vxrd = data.v[-1].cross(Motion(data.v[-1].linear,O3)).linear

r6dot = Motion(iv.linear,O3)
r6ddot = ia + iv.cross(r6dot)

qp = pinocchio.integrate(model,q,vq*dt)
pinocchio.forwardKinematics(model,data,qp,vqp,aqp)

Rp = data.oMi[-1].rotation
dR = R.T*Rp
rdotp  = dR*data.v[-1].linear
rddotp = dR*data.a[-1].linear + dR*cross(data.v[-1].angular,data.v[-1].linear)
vxrdp = dR*(data.v[-1].cross(Motion(data.v[-1].linear,O3)).linear)

assert( absmax( (rdotp-rdot)/dt - rddot ) < 1e-5)

'''
