from crocoddyl import StateVector
import numpy as np
from numpy.random import rand
from crocoddyl import IntegratedActionModelRK4
from crocoddyl import DifferentialActionModelLQR
from crocoddyl import a2m, m2a
from numpy.linalg import norm
from testutils import df_dx

np.set_printoptions(linewidth=np.nan, suppress=True)
#--------------frcom scipy.stats.ortho_group----------

#-------------------------------------------------------------------------------


nq = 10; nu = 5
nv = nq


dmodel = DifferentialActionModelLQR(nq, nu, driftFree=False)
ddata  = dmodel.createData()
model  = IntegratedActionModelRK4(dmodel)
data   = model.createData()

x = model.State.rand()

#u = np.random.rand( model.nu )
u = rand(model.nu)
xn,c = model.calc(data,x,u)

model.timeStep = 1

from crocoddyl import ActionModelNumDiff
mnum = ActionModelNumDiff(model,withGaussApprox=False)
dnum = mnum.createData()

model.calcDiff(data,x,u)


def get_k(q,v):
  x_ = np.vstack([q,v])
  model.calc(data,m2a(x_),u)
  return [a2m(ki) for ki in data.ki]

def get_ku(u):
  model.calc(data,x,m2a(u))
  return [a2m(ki) for ki in data.ki]

def get_xn(u):
  model.calc(data,x,m2a(u))
  return a2m(data.xnext)#.copy()

def get_au(u):
  a, l = model.differential.calc(data.differential[0],x,m2a(u))
  return a2m(a)

def get_y(q,v):
  x_ = np.vstack([q,v])
  model.calc(data,m2a(x_),u)
  return [a2m(y) for y in data.y]


dxn_du = df_dx(lambda _u: get_xn(_u), a2m(u))

dk_du = lambda i: df_dx(lambda _u: get_ku(_u)[i],
                        a2m(u))

dk_dq = lambda i: df_dx(lambda _q: get_k(_q,a2m(x[nq:]))[i],
                        a2m(x[:nq]))

dk_dv = lambda i: df_dx(lambda _v: get_k(a2m(x[:nq]), _v)[i],
                        a2m(x[nq:]))

dy_dq = lambda i: df_dx(lambda _q: get_y(_q, a2m(x[nq:]))[i],
                        a2m(x[:nq]))

dy_dv = lambda i: df_dx(lambda _v: get_y(a2m(x[:nq]), _v)[i],
                        a2m(x[nq:]))

e_k = lambda i: data.dki_dx[i][:,:nv]- dk_dq(i)



assert(np.allclose(data.Fu, dxn_du, atol=1e4*mnum.disturbance))

for i in xrange(4):
  assert(np.allclose(data.dki_du[i], dk_du(i), atol=1e4*mnum.disturbance))

for i in xrange(4):
  assert(np.allclose(data.dki_dx[i][:, :nv], dk_dq(i), atol=1e4*mnum.disturbance))
  assert(np.allclose(data.dki_dx[i][:, nv:], dk_dv(i), atol=1e4*mnum.disturbance))

for i in xrange(4):
  assert(np.allclose(data.dy_dx[i][:, :nv], dy_dq(i), atol=1e4*mnum.disturbance))
  assert(np.allclose(data.dy_dx[i][:, nv:], dy_dv(i), atol=1e4*mnum.disturbance))

mnum.calcDiff(dnum,x,u)
assert(np.allclose(data.Fx, dnum.Fx, atol=1e2*mnum.disturbance))
assert(np.allclose(data.Fu, dnum.Fu, atol=1e2*mnum.disturbance))
assert(np.allclose(data.Lu, dnum.Lu, atol=1e4*mnum.disturbance))
assert(np.allclose(data.Lx, dnum.Lx, atol=1e4*mnum.disturbance))

def get_attr_analytical(x,u,attr):
  _u = m2a(u)
  _x = m2a(x)
  model.calcDiff(data,_x,_u)
  return a2m(getattr(data, attr))#.copy()

eps = mnum.disturbance
Lxx0 = df_dx(lambda _x: get_attr_analytical(_x,u, "Lx"), a2m(x), h=eps)

Lxu0 = df_dx(lambda _u: get_attr_analytical(x,_u, "Lx"), a2m(u), h=eps)

Luu0 = df_dx(lambda _u: get_attr_analytical(x,_u, "Lu"), a2m(u), h=eps)

assert(np.allclose(Lxx0, data.Lxx, atol=1e4*mnum.disturbance))
assert(np.allclose(Lxu0, data.Lxu, atol=1e4*mnum.disturbance))
assert(np.allclose(Luu0, data.Luu, atol=1e4*mnum.disturbance))
