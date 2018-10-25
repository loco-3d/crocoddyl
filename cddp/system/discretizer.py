import abc
import numpy as np

class DiscretizerBase(object):
  """ This abstract class declares the virtual method for any discretization
  method of system dynamics.
  """
  __metaclass__=abc.ABCMeta
  
  @abc.abstractmethod
  def __init__(self):
    return
  """
  @abc.abstractmethod
  def __call__(ddpModel, ddpIData):
    pass
  """
  class fx(object):
    """ Abstract Class for the derivative for the function wrt x """
    
    __metaclass__=abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, dimf, dimx):
      pass
    """
    @abc.abstractmethod
    def multiply(Vxx, self):
      pass

    @abc.abstractmethod
    def transposemultiply(self, Vxx):
      pass
    """

  class fu(object):
    """ Abstract Class for the derivative for the function wrt u """
    
    __metaclass__=abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, dimf, dimx):
      pass
    """
    @abc.abstractmethod
    def multiply(Vxx, self):
      pass

    @abc.abstractmethod
    def transposemultiply(self, Vxx):
      pass
    """
class FloatingBaseMultibodyEulerDiscretizer(DiscretizerBase):
  """ Convert the time-continuos dynamics into time-discrete one by using
    forward Euler rule."""
  def __init__(self):
    return

  @staticmethod
  def __multiply__(fx, ):
    ddpIData.dynamicsData.fx
    return

  class fx(DiscretizerBase.fx):

    def __init__(self, dimq, dimv, dimf, dimx):
      self.aq = np.empty((dimv, dimq)) #derivative of ddq wrt q
      self.av = np.empty((dimv, dimv)) #derivative of ddq wrt v
    """
    @abc.abstractmethod
    def multiply(dt, V, self):
      pass

    @abc.abstractmethod
    def transposemultiply(self, Vxx):
    pass
    """
  class fu(object):
    
    def __init__(self, dimv, dimu, dimf):
      self.au = np.empty((dimv, dimu)) #derivative of ddq wrt u
      return
    """
    @abc.abstractmethod
    def multiply(V, self):
      pass

    @abc.abstractmethod
    def transposemultiply(self, Vxx):
      pass
    """
