import numpy as np
from itertools import izip


class DDPTerminalIntervalData(object):
  """ Data structure for a terminal interval of the DDP data.

  We create data for the nominal and new state values.
  """
  def __init__(self, ddpModel, tFinal):
    """ Constructs the data of the terminal interval

    :param ddpModel: DDP model
    :param tFinal: final time of the internal
    """
    # Dynamic and cost data of the terminal interval
    self.dynamicsData = ddpModel.createTerminalDynamicsData(tFinal)
    self.costData = ddpModel.createTerminalCostData()

    # Value function derivatives of the terminal interval
    self.Vx = np.zeros((ddpModel.dynamicsModel.nx(), 1))
    self.Vxx = np.zeros((ddpModel.dynamicsModel.nx(),ddpModel.dynamicsModel.nx()))


class DDPRunningIntervalData(object):
  """ Data structure for a running interval of the DDP data.

  We create data for the nominal and new state and control values. Additionally,
  this data structure contains regularized terms too (e.g. Quu_r).
  """
  def __init__(self, ddpModel, tInit, tFinal):
    """ Constructs the data of the running interval

    :param ddpModel: DDP model
    :param tInit: initial time of the internal
    :param tFinal: final time of the internal
    """
    # Dynamic and cost data of the running interval
    self.dynamicsData = ddpModel.createRunningDynamicsData(tInit, tFinal - tInit)
    self.costData = ddpModel.createRunningCostData()

    # Value function derivatives of the running interval
    self.Vx = np.zeros((ddpModel.dynamicsModel.nx(), 1))
    self.Vxx = np.zeros((ddpModel.dynamicsModel.nx(),ddpModel.dynamicsModel.nx()))

    # Quadratic approximation of the Value function
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

    # Feedback and feed-forward terms
    self.K = np.zeros((ddpModel.dynamicsModel.nu(), ddpModel.dynamicsModel.nx()))
    self.j = np.zeros((ddpModel.dynamicsModel.nu(), 1))

    # Extra DDP terms
    self.jt_Quu_j = 0.
    self.jt_Qu = 0.


class DDPData(object):
  """ Data structure for all time intervals of the DDP.
  """
  def __init__(self, ddpModel, timeline):
    """ Constructs the data for all DDP time intervals

    :param ddpModel: DDP model
    :param timeline: time vector
    """
    # Time, horizon and number of iterations
    self.timeline = timeline
    self.N = len(timeline) - 1
    self.n_iter = -1.    

    # Interval data for the cost and dynamics, and their quadratic and linear
    # approximation, respectively
    self.intervalDataVector = [DDPRunningIntervalData(ddpModel, timeline[i], timeline[i+1])
                               for i in xrange(self.N)]
    self.intervalDataVector.append(DDPTerminalIntervalData(ddpModel, timeline[-1]))

    # Cost-related data
    self.totalCost = 0.
    self.totalCost_prev = 0.
    self.dV_exp = 0.
    self.dV = 0.

    # Line search and regularization data
    self.alpha = -1.
    self.muLM = -1.
    self.muV = -1.
    self.muI = np.zeros((ddpModel.dynamicsModel.nu(),ddpModel.dynamicsModel.nu()))

    # Convergence and stopping criteria data
    self._convergence = False
    self.gamma = 0.
    self.theta = 0.
    self.z_new = 0.
    self.z = 0.


class DDPModel(object):
  """ Class to save the model information for the system, cost and dynamics
  """
  def __init__(self, dynamicsModel, costManager):
    self.dynamicsModel = dynamicsModel
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
    # Copying the current state into the previous one
    np.copyto(ddpData.dynamicsData.x_prev, ddpData.dynamicsData.x)
    self.dynamicsModel.backwardTerminalCalc(ddpData.dynamicsData)
    self.costManager.backwardTerminalCalc(ddpData.costData, ddpData.dynamicsData)

  def backwardRunningCalc(self, ddpData):
    """Performs the calculations before the backward pass"""
    # Copying the current state and control into the previous ones
    np.copyto(ddpData.dynamicsData.x_prev, ddpData.dynamicsData.x)
    np.copyto(ddpData.dynamicsData.u_prev, ddpData.dynamicsData.u)
    self.dynamicsModel.backwardRunningCalc(ddpData.dynamicsData)
    self.costManager.backwardRunningCalc(ddpData.costData, ddpData.dynamicsData)

  def createRunningDynamicsData(self, tInit, dt):
    return self.dynamicsModel.createData(tInit, dt)

  def createRunningCostData(self):
    return self.costManager.createRunningData(self.dynamicsModel.nx(),
                                              self.dynamicsModel.nu())

  def createTerminalDynamicsData(self, tFinal):
    return self.dynamicsModel.createData(tFinal, 0.)

  def createTerminalCostData(self):
    return self.costManager.createTerminalData(self.dynamicsModel.nx())

  def setInitial(self, ddpData, xInit, UInit):
    """
    Performs data copying from init values to ddpData.
    """
    np.copyto(ddpData.intervalDataVector[0].dynamicsData.x, xInit)
    #np.copyto(ddpData.intervalDataVector[0].dynamicsData.x_new, xInit)
    for u, intervalData in izip(UInit,ddpData.intervalDataVector[:-1]):
      np.copyto(intervalData.dynamicsData.u, u)
    return

  def setRunningReference(self, ddpData, Xref, name):
    index = self.costManager.getRunningCostIndex(name)
    for k, data in enumerate(ddpData.intervalDataVector[:-1]):
      self.costManager.runningCosts[index].setReference(data.costData.costVector[index], Xref[k])

  def setTerminalReference(self, ddpData, xref, name):
    index = self.costManager.getTerminalCostIndex(name)
    self.costManager.terminalCosts[index].setReference(
      ddpData.intervalDataVector[-1].costData.costVector[index], xref)