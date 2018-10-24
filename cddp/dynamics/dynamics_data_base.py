import abc

class DynamicsDataBase(object):
  "Base class to define interface for Dynamics"
  __metaclass__=abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, ddpModel):
    pass

  @abc.abstractmethod
  def forwardRunningCalc(self):
    "implement compute all terms for forward pass"
    pass

  @abc.abstractmethod
  def forwardTerminalCalc(self):
    "implement compute all terms for forward pass"
    pass
  
  @abc.abstractmethod
  def backwardRunningCalc(self):
    "implement compute all terms for backward pass"
    pass

  @abc.abstractmethod
  def backwardTerminalCalc(self):
    "implement compute all terms for backward pass"
    pass
