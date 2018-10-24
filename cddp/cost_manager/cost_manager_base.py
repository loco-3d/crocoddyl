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

  def __init__(self,dynamicsModel):
    self.dynamicsModel = dynamicsModel
    self.runningCosts = []
    self.terminalCosts = []
    pass

  @abc.abstractmethod
  def createRunningData(self, ddpModel):
    pass

  @abc.abstractmethod
  def createTerminalData(self, ddpModel):
    pass
