import abc

class DynamicsData(object):
  "Base class to define interface for Dynamics"
  __metaclass__=abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, ddpModel):
    pass


class DynamicsModel(object):
  "Base class to define the dynamics model"
  __metaclass__ = abc.ABCMeta
  
  def __init__(self, nxImpl, nx, nu):
    self._nx_impl = nxImpl
    self._nx = nx
    self._nu = nu

  @staticmethod
  def forwardRunningCalc(dynamicsModel, dynamicsData):
    "implement compute all terms for forward pass"
    pass

  @staticmethod
  def forwardTerminalCalc(dynamicsModel, dynamicsData):
    "implement compute all terms for forward pass"
    pass

  @staticmethod
  def backwardRunningCalc(dynamicsModel, dynamicsData):
    "implement compute all terms for backward pass"
    pass

  @staticmethod
  def backwardTerminalCalc(dynamicsModel, dynamicsData):
    "implement compute all terms for backward pass"
    pass

  @abc.abstractmethod
  def createData(self):
    pass

  @abc.abstractmethod
  def deltaX(self, x0, x1):
    pass

  def nxImpl(self):
    return self._nx_impl

  def nx(self):
    return self._nx

  def nu(self):
    return self._nu