import abc

class DynamicsData(object):
  "Base class to define interface for Dynamics"
  __metaclass__=abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, ddpModel):
    pass


class DynamicsModel(object):
  "Base class to define the dynamics model"
  __metaclass__=abc.ABCMeta
  @abc.abstractmethod
  def __init__(self):
    pass
  
  @abc.abstractmethod
  def createData(self):
    pass

  @abc.abstractmethod
  def forwardRunningCalc(self, dynamicsData):
    "implement compute all terms for forward pass"
    pass

  @abc.abstractmethod
  def forwardTerminalCalc(self, dynamicsData):
    "implement compute all terms for forward pass"
    pass
  
  @abc.abstractmethod
  def backwardRunningCalc(self, dynamicsData):
    "implement compute all terms for backward pass"
    pass

  @abc.abstractmethod
  def backwardTerminalCalc(self, dynamicsData):
    "implement compute all terms for backward pass"
    pass

  @abc.abstractmethod
  def deltaX(self, x0, x1):
    pass

  @abc.abstractmethod
  def nx(self):
    pass

  @abc.abstractmethod
  def nxImpl(self):
    pass
  
  @abc.abstractmethod
  def nu(self):
    pass