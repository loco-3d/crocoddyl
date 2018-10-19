import abc

class CostManagerIntervalDataBase(object):

  __metaclass__=abc.ABCMeta

  @abc.abstractmethod
  def __init__(self):
    pass

  @abc.abstractmethod
  def forwardRunningCalc(self, dynamicsModel, dynamicsData):
    pass

  @abc.abstractmethod
  def forwardTerminalCalc(self, dynamicsModel, dynamicsData):
    pass  
  
class CostManagerBase(object):

  __metaclass__=abc.ABCMeta

  def __init__(self):
    self.runningCosts = []
    self.terminalCosts = []
    pass

  @abc.abstractmethod
  def createRunningIntervalData(self):
    pass

  @abc.abstractmethod
  def createTerminalIntervalData(self):
    pass
  
