from crocoddyl import StateVector
import numpy as np
from numpy.random import rand
from crocoddyl import IntegratedActionDataRK4, IntegratedActionModelRK4
from crocoddyl import DifferentialActionModelLQR, DifferentialActionDataLQR
from crocoddyl import a2m, m2a
from pinocchio.utils import zero
from numpy.linalg import norm

np.set_printoptions(linewidth=np.nan, suppress=True)
#--------------frcom scipy.stats.ortho_group----------

#-------------------------------------------------------------------------------


def df_dx(func,v,h=1e-9):
  dv = zero(v.size)
  f0 = func(v)
  res = np.zeros([len(f0),v.size])
  for iv in range(v.size):
    dv[iv] = h
    res[:,iv] = (func(v+dv) - f0)/h
    dv[iv] = 0
  return res


nq = 10; nu = 10
nv = nq


dmodel = DifferentialActionModelLQR(nq, nu)
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
  return data.ki

def get_ku(u):
  model.calc(data,x,m2a(u))
  return data.ki

def get_xn(u):
  model.calc(data,x,m2a(u))
  return data.xnext.copy()

def get_au(u):
  a, l = model.differential.calc(data.differential[0],x,m2a(u))
  return a

def get_y(q,v):
  x_ = np.vstack([q,v])
  model.calc(data,m2a(x_),u)
  return data.y


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


tolerance = 1e-4
assert(np.isclose(data.Fu, dxn_du, atol=tolerance).all())

for i in xrange(4):
  assert(np.isclose(data.dki_du[i], dk_du(i), atol=tolerance).all())

for i in xrange(4):
  assert(np.isclose(data.dki_dx[i][:,:nv], dk_dq(i), atol=tolerance).all())
  assert(np.isclose(data.dki_dx[i][:,nv:], dk_dv(i), atol=tolerance).all())

for i in xrange(4):
  assert(np.isclose(data.dy_dx[i][:,:nv], dy_dq(i), atol=tolerance).all())
  assert(np.isclose(data.dy_dx[i][:,nv:], dy_dv(i), atol=tolerance).all())


mnum.calcDiff(dnum,x,u)
assert( norm(data.Fx-dnum.Fx) < 1e-8 )
assert( norm(data.Fu-dnum.Fu) < 1e-8 )
assert( norm(data.Lu-dnum.Lu) < 10*np.sqrt(mnum.disturbance) )
assert( norm(data.Lx-dnum.Lx) < 10*np.sqrt(mnum.disturbance) )


def get_attr_analytical(x,u,attr):
  _u = m2a(u)
  _x = m2a(x)
  model.calcDiff(data,_x,_u)
  return getattr(data, attr).copy()

Lxx0 = df_dx(lambda _x: get_attr_analytical(_x,u, "Lx"), a2m(x))

Lxu0 = df_dx(lambda _u: get_attr_analytical(x,_u, "Lx"), a2m(u))

Luu0 = df_dx(lambda _u: get_attr_analytical(x,_u, "Lu"), a2m(u))


assert( norm(Lxx0-data.Lxx) < 1e-5)#*np.sqrt(mnum.disturbance) )
assert( norm(Lxu0-data.Lxu) < 1e-5)#10*np.sqrt(mnum.disturbance) )
assert( norm(Luu0-data.Luu) < 1e-5)#10*mnum.disturbance )
