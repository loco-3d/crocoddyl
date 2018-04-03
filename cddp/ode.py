from abc import ABCMeta, abstractmethod


class ODEBase:
  __metaclass__ = ABCMeta

  def __init__(self,n):
    """
    Constructor of Ordinary Differential Equation
    :param n: dimension of the ODE
    """
    self.n = n

  @abstractmethod
  def f(self,x):
    pass