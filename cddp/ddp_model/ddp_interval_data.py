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
    
    self.dynamicsData = ddpModel.dynamicsModel.createIntervalData(tFinal)
    self.costData = ddpModel.costManager.createRunningIntervalData()


  def forwardCalc(self):
    """Performes the dynamics integration to generate the state and control functions"""
    self.dynamicsData.forwardTerminalCalc()
    self.costData.forwardTerminalCalc(self.ddpModel.dynamicsModel, self.dynamicsData);
    pass

  def backwardCalc(self):
    """Performs the calculations before the backward pass
    Pinocchio Data has already been filled with the forward pass."""

    #Do Not Change The Order
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

    self.dynamicsData = ddpModel.dynamicsModel.createIntervalData(tInit)
    self.costData = ddpModel.costManager.createRunningIntervalData()

  def forwardCalc(self):
    """Performes the dynamics integration to generate the state and control functions"""
    self.dynamicsData.forwardRunningCalc()
    self.costData.forwardRunningCalc(self.ddpModel.dynamicsModel, self.dynamicsData);

  def backwardCalc(self):
    """Performs the calculations before the backward pass"""
    self.costData.backwardRunningCalc(self.ddpModel.dynamicsModel, self.dynamicsData)
    self.dynamicsData.backwardRunningCalc()
    
    # Feedback and feedforward terms
    #self.K = np.matrix(np.zeros((self.system.m, self.system.ndx)))
    #self.j = np.matrix(np.zeros((self.system.m, 1)))

    # Value function and its derivatives
    #self.Vx = np.matrix(np.zeros((self.system.ndx, 1)))
    #self.Vxx = np.matrix(np.zeros((self.system.ndx, self.system.ndx)))

    # Quadratic approximation of the value function
    # self.Qx = np.matrix(np.zeros((self.system.ndx, 1)))
    # self.Qu = np.matrix(np.zeros((self.system.m, 1)))
    # self.Qxx = np.matrix(np.zeros((self.system.ndx, self.system.ndx)))
    # self.Qux = np.matrix(np.zeros((self.system.m, self.system.ndx)))
    # self.Quu = np.matrix(np.zeros((self.system.m, self.system.m)))
    # self.Qux_r = np.matrix(np.zeros((self.system.m, self.system.ndx)))
    # self.Quu_r = np.matrix(np.zeros((self.system.m, self.system.m)))
