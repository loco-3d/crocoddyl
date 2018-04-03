from abc import ABCMeta, abstractmethod
import numpy as np


class RunningCostBase:
  __metaclass__ = ABCMeta

  def __init__(self,model):
    self.model = model
    self.n = model.n
    self.m = model.m
    pass

  @abstractmethod
  def createData(self,model_data):
    pass

  @abstractmethod
  def l(self,x,u,data):
    pass

  @abstractmethod
  def l_x(self,x,u,data):
    pass

  @abstractmethod
  def l_u(self,x,u,data):
    pass

  @abstractmethod
  def l_xx(self,x,u,data):
    pass

  @abstractmethod
  def l_xu(self,x,u,data):
    pass

  @abstractmethod
  def l_uu(self,x,u,data):
    pass

class RunningCostDataBase:
  __metaclass__ = ABCMeta

  def __init__(self, model, model_data):
    self.model_data = model_data
    self.n = model.n
    self.m = model.m

    # Value of the error
    self.m_cost_value = float('Inf')
    self.m_lx = np.matrix([float('Inf')] * self.n).T
    self.m_lu = np.matrix([float('Inf')] * self.m).T
    self.m_lxx = np.matrix(np.zeros((self.n,self.n)))
    self.m_lxx.fill(float('Inf'))
    self.m_lxu = np.matrix(np.zeros((self.n, self.m)))
    self.m_lxu.fill(float('Inf'))

    self.m_luu = np.matrix(np.zeros((self.m, self.m)))
    self.m_luu.fill(float('Inf'))


class TerminalCostBase:
  __metaclass__ = ABCMeta

  def __init__(self,model):
    self.model = model
    self.n = model.n
    pass

  @abstractmethod
  def createData(self):
    pass

  @abstractmethod
  def l(self,x,data):
    pass

  @abstractmethod
  def l_x(self,x,data):
    pass

  @abstractmethod
  def l_xx(self,x,data):
    pass

class TerminalCostDataBase:
  __metaclass__ = ABCMeta

  def __init__(self, model, model_data):
    self.model_data = model_data
    self.n = model.n

    # Value of the error
    self.m_cost_value = float('Inf')
    self.m_lx = np.matrix([float('Inf')] * self.n).T
    self.m_lxx = np.matrix(np.zeros((self.n,self.n)))
    self.m_lxx.fill(float('Inf'))


class SumOfRunningCost(RunningCostBase):

  def __init__(self,model):
    RunningCostBase.__init__(self,model)

    self.weigths = []
    self.cost_functions = []

  def createData(self,model_data):
    data = SumOfRunningCostData(self.model,model_data)

    for cost in self.cost_functions:
      data.cost_data.append(cost.createData(model_data))

    return data

  def append(self,cost,weigth=1.):
    self.cost_functions.append(cost)
    self.weigths.append(weigth)

  @staticmethod
  def __call_method__(entity,name,x,u,data):
    method = getattr(entity,name)
    return method(x,u,data)

  def l(self,x,u,data):
    cost_value = 0.
    for k,cost in enumerate(self.cost_functions):
      cost_data = data.cost_data[k]
      w = self.weigths[k]
      cost_value += w * cost.l(x,u,cost_data)

    data.m_cost_value = cost_value
    return cost_value

  def l_x(self,x,u,data):
    m_lx = data.m_lx
    m_lx.fill(0.)

    for k,cost in enumerate(self.cost_functions):
      cost_data = data.cost_data[k]
      w = self.weigths[k]
      m_lx += w * cost.l_x(x,u,cost_data)

    return m_lx

  def l_u(self,x,u,data):
    m_lu = data.m_lu
    m_lu.fill(0.)

    for k,cost in enumerate(self.cost_functions):
      cost_data = data.cost_data[k]
      w = self.weigths[k]
      m_lu += w * cost.l_u(x,u,cost_data)

    return m_lu

  def l_uu(self,x,u,data):
    m_luu = data.m_luu
    m_luu.fill(0.)

    for k,cost in enumerate(self.cost_functions):
      cost_data = data.cost_data[k]
      w = self.weigths[k]
      m_luu += w * cost.l_uu(x,u,cost_data)

    return m_luu

  def l_xu(self,x,u,data):
    m_lxu = data.m_lxu
    m_lxu.fill(0.)


    for k,cost in enumerate(self.cost_functions):
      cost_data = data.cost_data[k]
      w = self.weigths[k]
      m_lxu += w * cost.l_xu(x,u,cost_data)

    return m_lxu

  def l_xx(self,x,u,data):
    m_lxx = data.m_lxx
    m_lxx.fill(0.)

    for k,cost in enumerate(self.cost_functions):
      cost_data = data.cost_data[k]
      w = self.weigths[k]
      m_lxx += w * cost.l_xx(x,u,cost_data)

    return m_lxx


class SumOfRunningCostData(RunningCostDataBase):

  def __init__(self, model, model_data):
    RunningCostDataBase.__init__(self,model,model_data)
    self.cost_data = []
