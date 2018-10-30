import numpy as np


class DDPIntervalData(object):
  """ Define the base class for data element for an interval."""
  def __init__(self, ddpModel):
    self.Vx = np.zeros((ddpModel.dynamicsModel.nx(), 1))
    self.Vxx = np.zeros((ddpModel.dynamicsModel.nx(),ddpModel.dynamicsModel.nx()))


class TerminalDDPData(DDPIntervalData):
  """ Data structure for the terminal interval of the DDP.

  We create data for the nominal and new state values.
  """
  def __init__(self, ddpModel, tFinal):
    DDPIntervalData.__init__(self, ddpModel)
    self.tFinal = tFinal

    self.dynamicsData = ddpModel.createTerminalDynamicsData(tFinal)
    self.costData = ddpModel.createTerminalCostData()


class RunningDDPData(DDPIntervalData):
  """ Data structure for the running interval of the DDP.

  We create data for the nominal and new state and control values. Additionally,
  this data structure contains regularized terms too (e.g. Quu_r).
  """

  def __init__(self, ddpModel, tInit, tFinal):
    DDPIntervalData.__init__(self, ddpModel)

    self.tInit = tInit
    self.tFinal = tFinal
    self.dt = self.tFinal-self.tInit

    self.dynamicsData = ddpModel.createRunningDynamicsData(tInit)
    self.costData = ddpModel.createRunningCostData()

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


class DDPData(object):
  """ Base class to define the structure for storing and accessing data elements at each 
  DDP interval
  """

  def __init__(self, ddpModel, timeline):
    """Initializing the data elements"""

    self.timeline = timeline
    self.ddpModel = ddpModel
    self.N = len(timeline) - 1
    self.intervalDataVector = [RunningDDPData(ddpModel, timeline[i], timeline[i+1])
                               for i in xrange(self.N)]
    self.intervalDataVector.append(TerminalDDPData(ddpModel, timeline[-1]))

    #Total Cost
    self.totalCost = 0.
    self.totalCost_prev = 0.
    self.dV_exp = 0.
    self.dV = 0.
    
    #Run time variables
    self._convergence = False
    self.muLM = -1.
    self.muV = -1.
    self.muI = np.zeros((self.ddpModel.dynamicsModel.nu(),self.ddpModel.dynamicsModel.nu()))
    self.alpha = -1.
    self.n_iter = -1.
    
    #Analysis Variables
    self.gamma = 0.
    self.theta = 0.
    self.z_new = 0.
    self.z = 0.


class DDPModel(object):
  """ Class to save the model information for the system, cost and dynamics
  """
  def __init__(self, dynamicsModel, integrator, discretizer, costManager):
    self.dynamicsModel = dynamicsModel
    self.integrator = integrator
    self.discretizer = discretizer
    self.costManager = costManager

  def forwardTerminalCalc(self, ddpData):
    """Performes the dynamics integration to generate the state and control functions"""
    self.dynamicsModel.forwardTerminalCalc(ddpData.dynamicsData)
    self.costManager.forwardTerminalCalc(ddpData.costData, ddpData.dynamicsData)

  def forwardRunningCalc(self, ddpData):
    """Performes the dynamics integration to generate the state and control functions"""
    self.dynamicsModel.forwardRunningCalc(ddpData.dynamicsData)
    self.costManager.forwardRunningCalc(ddpData.costData, ddpData.dynamicsData)

  def backwardTerminalCalc(self, ddpData):
    """Performs the calculations before the backward pass
    Pinocchio Data has already been filled with the forward pass."""
    self.dynamicsModel.backwardTerminalCalc(ddpData.dynamicsData)
    self.costManager.backwardTerminalCalc(ddpData.costData, ddpData.dynamicsData)

  def backwardRunningCalc(self, ddpData):
    """Performs the calculations before the backward pass"""
    self.dynamicsModel.backwardRunningCalc(ddpData.dynamicsData)
    self.costManager.backwardRunningCalc(ddpData.costData, ddpData.dynamicsData)

  def createRunningDynamicsData(self, tInit):
    return self.dynamicsModel.createData(self, tInit)

  def createRunningCostData(self):
    return self.costManager.createRunningData(self.dynamicsModel)

  def createTerminalDynamicsData(self, tFinal):
    return self.dynamicsModel.createData(self, tFinal)

  def createTerminalCostData(self):
    return self.costManager.createTerminalData(self.dynamicsModel)
