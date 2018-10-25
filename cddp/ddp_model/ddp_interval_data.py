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

    self.Vx = np.zeros((ddpModel.dynamicsModel.nx(), 1))
    self.Vxx = np.zeros((ddpModel.dynamicsModel.nx(),ddpModel.dynamicsModel.nx()))

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

    self.dynamicsData = ddpModel.createRunningDynamicsData(tInit)
    self.costData = ddpModel.createRunningCostData()

    self.Vx = np.zeros((ddpModel.dynamicsModel.nx(), 1))
    self.Vxx = np.zeros((ddpModel.dynamicsModel.nx(),ddpModel.dynamicsModel.nx()))

    self.Qx = np.zeros((ddpModel.dynamicsModel.nx(), 1))
    self.Qu = np.zeros((ddpModel.dynamicsModel.nu(), 1))
    self.Qxx = np.zeros((ddpModel.dynamicsModel.nx(),ddpModel.dynamicsModel.nx()))
    self.Qux = np.zeros((ddpModel.dynamicsModel.nu(),ddpModel.dynamicsModel.nx()))
    self.Quu = np.zeros((ddpModel.dynamicsModel.nu(),ddpModel.dynamicsModel.nu()))
    
    self.Quu_r = np.zeros((ddpModel.dynamicsModel.nu(),ddpModel.dynamicsModel.nu()))
    self.Qux_r = np.zeros((ddpModel.dynamicsModel.nu(),ddpModel.dynamicsModel.nx()))
    self.L = np.zeros((ddpModel.dynamicsModel.nu(),ddpModel.dynamicsModel.nu()))
    self.L_inv = np.zeros((ddpModel.dynamicsModel.nu(),ddpModel.dynamicsModel.nu()))
    self.Quu_inv_minus = np.zeros((ddpModel.dynamicsModel.nu(),ddpModel.dynamicsModel.nu()))

    self.K = np.zeros((ddpModel.dynamicsModel.nu(), ddpModel.dynamicsModel.nx()))
    self.j = np.zeros((ddpModel.dynamicsModel.nu(), 1))

    self.jt_Quu_j = 0.
    self.jt_Qu = 0.
    
  def forwardCalc(self):
    """Performes the dynamics integration to generate the state and control functions"""
    self.dynamicsData.forwardRunningCalc()
    self.costData.forwardRunningCalc(self.ddpModel.dynamicsModel, self.dynamicsData);

  def backwardCalc(self):
    """Performs the calculations before the backward pass"""
    self.costData.backwardRunningCalc(self.ddpModel.dynamicsModel, self.dynamicsData)
    self.dynamicsData.backwardRunningCalc()
