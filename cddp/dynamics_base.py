from abc import ABCMeta, abstractmethod
from ode import ODEBase


class DynamicModelBase(ODEBase):
  __metaclass__ = ABCMeta

  def __init__(self,n,m):
    '''
    :param n: dimension of the state
    :param m: dimensions of the control
    '''
    self.n = n
    self.m = m

  @abstractmethod
  def createData(self):
    pass

  @abstractmethod
  def f(self,x,u):
    '''
    Eval the current at the given state and control inputs and store the result in data
    :param x: current state of the system
    :param u: current control input
    :return: state variation
    '''
    pass


  @abstractmethod
  def f_x(self,x,u,data):
    '''
    Eval the Jacobian of the dynamics w.r.t. the state and store the result in data
    :param x: current state of the system
    :param u: current control input
    :return: Jacobian of the state variation w.r.t the state
    '''
    pass

  @abstractmethod
  def f_u(self, x, u, data):
    '''
    Eval the Jacobian of the dynamics w.r.t. the control and store the result in data
    :param x: current state of the system
    :param u: current control input
    :return: Jacobian of the state variation w.r.t the control
    '''
    pass

  @abstractmethod
  def f_xx(self, x, u, data):
    '''
    Eval the hessian of the dynamics with respect to the state
    :param x:
    :param u:
    :param data:
    :return:
    '''
    pass

  @abstractmethod
  def f_xu(self, x, u, data):
    pass

  @abstractmethod
  def f_uu(self, x, u, data):
    pass


class DynamicDataBase:
  __metaclass__ = ABCMeta

  def __init__(self):
    self.m_dx = None
    self.m_fx = None
    self.m_fu = None
    self.m_fxx = None
    self.m_fxu = None
    self.m_fuu = None
    pass