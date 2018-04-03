import numpy as np
from numpy.linalg import norm
from dynamics_base import *
from cost_base import *

## Dynamics model

class CentroidalDataReduced(DynamicDataBase):

  n = 9
  m = 6

  def __init__(self):


    self.m_dx = np.matrix(np.zeros((self.n,1)))
    self.m_fx = np.matrix(np.zeros((self.n,self.n)))
    self.m_fu = np.matrix(np.zeros((self.n,self.m)))

    self.m_fx[CentroidalModelReduced.com_position_id,CentroidalModelReduced.com_velocity_id] = np.eye(3)
    self.m_fu[CentroidalModelReduced.com_acceleration_id,CentroidalModelReduced.linear_control_id] = np.eye(3)
    self.m_fu[CentroidalModelReduced.am_variation_id,CentroidalModelReduced.angular_control_id] = np.eye(3)

class CentroidalModelReduced(DynamicModelBase):

  n = 9
  m = 6
  com_position_id = slice(0,3) # in state vector
  com_velocity_id = slice(3,6) # in state vector
  com_acceleration_id = slice(3,6) # in state variation

  am_id = slice(6,9) # in state vector
  am_variation_id = slice(6,9) # in state variation

  linear_control_id = slice(0,3) # in control vector
  angular_control_id = slice(3,6) # in control vector


  def __init__(self,mass):
    DynamicModelBase.__init__(self,n=self.n,m=self.m)
    self.mass = mass


  def createData(self):
    return CentroidalDataReduced()

  def f(self,x,u,data):
    data.m_dx[self.com_position_id] = x[self.com_velocity_id]
    data.m_dx[self.com_velocity_id] = u[self.linear_control_id]
    data.m_dx[self.am_id] = u[self.angular_control_id]

    return data.m_dx

  def f_x(self,x,u,data):
    return data.m_fx

  def f_u(self,x,u,data):
    return data.m_fu

  def f_xx(self, x, u, data):
    return data.m_fxx

  def f_xu(self, x, u, data):
    return data.m_fxu

  def f_uu(self,x,u,data):
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