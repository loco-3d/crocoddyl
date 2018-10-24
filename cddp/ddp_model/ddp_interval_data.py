import abc
import numpy as np

class DDPIntervalDataBase(object):
  """ Define the base class for data element for an interval."""
  __metaclass__=abc.ABCMeta

  def __init__(self, ddpModel):
    self.ddpModel = ddpModel

  @abc.abstractmethod
  def forwardCalc(self):
    pass
  
  @abc.abstractmethod
  def backwardCalc(self):
    pass

class TerminalDDPData(DDPIntervalDataBase):
  """ Data structure for the terminal interval of the DDP.

  We create data for the nominal and new state values.
  """
  def __init__(self, ddpModel, tFinal):
    DDPIntervalDataBase.__init__(self, ddpModel)
    self.tFinal = tFinal

    self.dynamicsData = ddpModel.createTerminalDynamicsData(tFinal)
    self.costData = ddpModel.createTerminalCostData()

  def forwardCalc(self):
    """Performes the dynamics integration to generate the state and control functions"""
    self.dynamicsData.forwardTerminalCalc()
    self.costData.forwardTerminalCalc(self.ddpModel.dynamicsModel, self.dynamicsData);
    pass

  def backwardCalc(self):
    """Performs the calculations before the backward pass
    Pinocchio Data has already been filled with the forward pass."""

    #Do Not Change The Order
    #print self.tFinal, "Final Backward Pass"
    self.costData.backwardTerminalCalc(self.ddpModel.dynamicsModel, self.dynamicsData)
    self.dynamicsData.backwardTerminalCalc()
   
class RunningDDPData(DDPIntervalDataBase):
  """ Data structure for the running interval of the DDP.

  We create data for the nominal and new state and control values. Additionally,
  this data structure contains regularized terms too (e.g. Quu_r).
  """

  def __init__(self, ddpModel, tInit, tFinal):
    DDPIntervalDataBase.__init__(self, ddpModel)

    self.tInit = tInit
    self.tFinal = tFinal

    self.dt = self.tFinal-self.tInit

    self.dynamicsData = ddpModel.createRunningDynamicsData(tInit)
    self.costData = ddpModel.createRunningCostData()

  def forwardCalc(self):
    """Performes the dynamics integration to generate the state and control functions"""
    self.dynamicsData.forwardRunningCalc()
    self.costData.forwardRunningCalc(self.ddpModel.dynamicsModel, self.dynamicsData);

  def backwardCalc(self):
    """Performs the calculations before the backward pass"""
    #print self.tInit, "Initial Backward Pass"    
    self.costData.backwardRunningCalc(self.ddpModel.dynamicsModel, self.dynamicsData)
    self.dynamicsData.backwardRunningCalc()
