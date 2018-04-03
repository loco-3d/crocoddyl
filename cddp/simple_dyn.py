import numpy as np
from numpy.linalg import norm
from dynamics_base import *
from cost_base import *
from math import sin, cos

## Dynamics model

class SimpleDynamicsData(DynamicDataBase):

  n = 1
  m = 1

  def __init__(self):

    self.m_dx = np.matrix(np.zeros((self.n,1)))
    self.m_fx = np.matrix(np.zeros((self.n,self.n)))
    self.m_fu = np.matrix(np.zeros((self.n,self.m)))

class SimpleDynamicsModel(DynamicModelBase):

  n = 1
  m = 1

  def __init__(self,dt):
    DynamicModelBase.__init__(self,n=self.n,m=self.m)
    self.dt = dt

  def createData(self):
    return SimpleDynamicsData()

  def f(self,x,u,data):

    data.m_dx = u[0]
    return data.m_dx

  def f_x(self,x,u,data):
    data.m_fx[:] = 0.
    return data.m_fx

  def f_u(self,x,u,data):
    data.m_fu[:] = 1.
    return data.m_fu

  def f_xx(self, x, u, data):
    data.m_fxx = 0.
    return data.m_fxx

  def f_xu(self, x, u, data):
    data.m_fxu = 0.
    return data.m_fxu

  def f_uu(self,x,u,data):
    data.m_fuu = 0.
    return data.m_fuu

## Terminal cost function

class FinalStateCostData(TerminalCostDataBase):

  def __init__(self,model,model_data):
    TerminalCostDataBase.__init__(self,model,model_data)
    # Value of the error
    self.m_e = np.matrix([float('Inf')]*self.n).T
    self.m_lx = self.m_e.copy()
    self.m_lxx = 2.*np.matrix(np.eye(self.n))

class FinalStateCost(TerminalCostBase):

  def __init__(self,model):
    TerminalCostBase.__init__(self,model)
    self.x_final = np.matrix(np.zeros((self.n,1)))

  def createData(self,model_data):
    return FinalStateCostData(self.model,model_data)

  def setFinalState(self,x_final):
    self.x_final[:] = x_final

  def l(self,x,data):
    data.m_e = x - self.x_final
    data.m_cost = float(data.m_e.T*data.m_e)

    return data.m_cost

  def l_x(self,x,data):
    data.m_e = x - self.x_final
    data.m_lx = 2.*data.m_e

    return data.m_lx

  def l_xx(self,x,data):
    return data.m_lxx

## Running least state cost function

class LeastStateCostData(RunningCostDataBase):
  def __init__(self, model, model_data):
    RunningCostDataBase.__init__(self, model, model_data)
    # Value of the error
    self.m_e = np.matrix([float('Inf')] * self.n).T
    self.m_lx = np.matrix(np.zeros((self.n,1)))
    self.m_lu = np.matrix(np.zeros((self.m,1)))
    self.m_lxu = np.matrix(np.zeros((self.n,self.m)))
    self.m_luu = np.matrix(np.zeros((self.m,self.m)))
    self.m_lxx = np.matrix(np.zeros((self.n,self.n)))
    self.m_lxx[:,:] = 2.*np.eye(self.n)

class LeastStateCost(RunningCostBase):
  def __init__(self, model):
    RunningCostBase.__init__(self, model)
    self.x_target = np.matrix(np.zeros((self.n, 1)))

  def createData(self, model_data):
    return LeastStateCostData(self.model, model_data)

  def setTargetState(self, x_target):
    self.x_target[:] = x_target

  def l(self, x, u, data):
    data.m_e = x - self.x_target
    data.m_cost = float(data.m_e.T * data.m_e)

    return data.m_cost

  def l_x(self, x, u, data):
    data.m_e = x - self.x_target
    data.m_lx = 2. * data.m_e

    return data.m_lx

  def l_u(self, x, u, data):
    return data.m_lu

  def l_xu(self, x, u, data):
    return data.m_lxu

  def l_uu(self, x, u, data):
    return data.m_luu

  def l_xx(self, x, u, data):
    return data.m_lxx

## Running cost function

class LeastControlCostData(RunningCostDataBase):

  def __init__(self,model,model_data):
    RunningCostDataBase.__init__(self,model,model_data)
    # Value of the error
    self.m_lx.fill(0.)
    self.m_lxx.fill(0.)
    self.m_lxu.fill(0.)


class LeastControlCost(RunningCostBase):

  def __init__(self,model):
    RunningCostBase.__init__(self,model)
    self.W = np.matrix(np.eye(self.m))

  def setW(self,R):
    assert R.shape == (self.m, self.m)
    assert np.isclose(R - R.T,np.matrix(np.zeros((self.m,self.m)))).all()

    self.R = R

  def createData(self,model_data):
    return LeastControlCostData(self.model,model_data)

  def l(self,x,u,data):
    data.m_cost = float(u.T*(self.R*u))

    return data.m_cost

  def l_u(self,x,u,data):
    data.m_lu = 2. * self.R * u
    return data.m_lu

  def l_uu(self,x,u,data):
    data.m_luu = 2. * self.R

    return data.m_luu

  def l_xu(self,x,u,data):
    return data.m_lxu

  def l_x(self,x,u,data):
    return data.m_lx

  def l_xx(self,x,u,data):
    return data.m_lxx