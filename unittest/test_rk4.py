from crocoddyl import StateVector
from crocoddyl import ActivationModelWeightedQuad
import numpy as np
from numpy.random import rand
from crocoddyl import IntegratedActionDataRK4, IntegratedActionModelRK4
import scipy as sp
from crocoddyl import a2m, m2a
from pinocchio.utils import zero
from numpy.linalg import norm

np.set_printoptions(linewidth=np.nan, suppress=True)
#--------------frcom scipy.stats.ortho_group----------

# Create a random orthonormal matrix using np.random.rand
def randomOrthonormalMatrix(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H


class DifferentialActionModelLQR:
  """
  This class implements a linear dynamics, and quadratic costs.
  Since the DAM is a second order system, and the integratedactionmodels are implemented
  as being second order integrators, This class implements a second order linear system
  given by
  x = [q, dq]
  
  ddq = A dq + B q + C u  ......A, B, C are constant
  
  Full dynamics:
  [dq] = [0  1][q]  +  [0]
  [ddq]  [B  A][dq] +  [C]u

  The cost function is given by l(x,u) = x^T*Q*x + u^T*U*u
  """

  def __init__(self,nq,nu):

    self.nq,self.nv = nq, nq
   
    self.nx = 2*self.nq
    self.ndx = self.nx
    self.nout = self.nv
    self.nu = nu
    self.unone = np.zeros(self.nu)
    self.State = StateVector(self.nx)
    self.nx = self.State.nx
    self.ndx = self.State.ndx
    self.nu = nu
    self.unone = np.zeros(self.nu)
    act = ActivationModelWeightedQuad(weights=np.array([2]*nv + [.5]*nv))

    v1 = randomOrthonormalMatrix(self.nq);
    v2 = randomOrthonormalMatrix(self.nq);
    v3 = randomOrthonormalMatrix(self.nq)
    e1 = rand(self.nq); e2 = rand(self.nq); e3 = rand(self.nq)

    self.Q = randomOrthonormalMatrix(self.nx);
    self.U = randomOrthonormalMatrix(self.nu)
    
    self.B = v1; self.A = v2; self.C = v3

    
  @property
  def ncost(self): return self.nx+self.nu
  def createData(self): return DifferentialActionDataLQR(self)
  def calc(model,data,x,u=None):
    q = x[:model.nq]; dq = x[model.nq:]
    data.xout[:] = (np.dot(model.A, dq) + np.dot(model.B, q) + np.dot(model.C, u)).flat
    data.cost = np.dot(x, np.dot(model.Q, x)) + np.dot(u, np.dot(model.U, u))
    return data.xout, data.cost
  
  def calcDiff(model,data,x,u=None,recalc=True):
    if u is None: u=model.unone
    if recalc: xout,cost = model.calc(data,x,u)
    
    data.Fx[:,:] = np.hstack([model.B, model.A])
    data.Fu[:,:]   = model.C
    data.Lx[:] = np.dot(x.T, data.Lxx)
    data.Lu[:] = np.dot(u.T, data.Luu)
    return data.xout,data.cost

class DifferentialActionDataLQR:
  def __init__(self,model):
    self.cost = np.nan
    self.xout = np.zeros(model.nout)
    nx,nu,ndx,nq,nv,nout = model.nx,model.nu,model.State.ndx,model.nq,model.nv,model.nout
    self.F = np.zeros([ nout,ndx+nu ])
    self.Fx = self.F[:,:ndx]
    self.Fu = self.F[:,-nu:]
    self.g = np.zeros( ndx+nu)
    self.L = np.zeros([ndx+nu,ndx+nu])
    self.Lx = self.g[:ndx]
    self.Lu = self.g[ndx:]
    self.Lxx = self.L[:ndx,:ndx]
    self.Lxu = self.L[:ndx,ndx:]
    self.Luu = self.L[ndx:,ndx:]

    self.Lxx = model.Q+model.Q.T
    self.Lxu = np.zeros((nx, nu))
    self.Luu = model.U+model.U.T

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
mnum = ActionModelNumDiff(model,withGaussApprox=True)
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
