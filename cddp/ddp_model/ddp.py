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
    # Current and previous state
    self.x = np.zeros((ddpModel.dynamicsModel.nxImpl(), 1))
    self.x_prev = np.zeros((ddpModel.dynamicsModel.nxImpl(), 1))

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
    # Current and previous state and control
    self.x = np.zeros((ddpModel.dynamicsModel.nxImpl(), 1))
    self.u = np.zeros((ddpModel.dynamicsModel.nu(), 1))
    self.x_prev = np.zeros((ddpModel.dynamicsModel.nxImpl(), 1))
    self.u_prev = np.zeros((ddpModel.dynamicsModel.nu(), 1))

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
    self.Vpxx_fx = np.zeros((ddpModel.dynamicsModel.nx(), ddpModel.dynamicsModel.nx()))
    self.Vpxx_fu = np.zeros((ddpModel.dynamicsModel.nx(), ddpModel.dynamicsModel.nu()))


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
    self.n_iter = -1

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

  def forwardTerminalCalc(self, dynamicsData, costData, x):
    """Performes the dynamics integration to generate the state and control functions"""
    self.dynamicsModel.forwardTerminalCalc(dynamicsData, x)
    self.costManager.forwardTerminalCalc(costData, dynamicsData, x)

  def forwardRunningCalc(self, dynamicsData, costData, x, u, xNext):
    """Performes the dynamics integration to generate the state and control functions"""
    self.dynamicsModel.forwardRunningCalc(dynamicsData, x, u, xNext)
    self.costManager.forwardRunningCalc(costData, dynamicsData, x, u)

  def backwardTerminalCalc(self, dynamicsData, costData, x):
    """Performs the calculations before the backward pass
    Pinocchio Data has already been filled with the forward pass."""
    self.dynamicsModel.backwardTerminalCalc(dynamicsData, x)
    self.costManager.backwardTerminalCalc(costData, dynamicsData, x)

  def backwardRunningCalc(self, dynamicsData, costData, x, u):
    """Performs the calculations before the backward pass"""
    self.dynamicsModel.backwardRunningCalc(dynamicsData, x, u)
    self.costManager.backwardRunningCalc(costData, dynamicsData, x, u)

  def createRunningDynamicsData(self, tInit, dt):
    return self.dynamicsModel.createData(tInit, dt)

  def createRunningCostData(self):
    return self.costManager.createRunningData(self.dynamicsModel)

  def createTerminalDynamicsData(self, tFinal):
    return self.dynamicsModel.createData(tFinal, 0.)

  def createTerminalCostData(self):
    return self.costManager.createTerminalData(self.dynamicsModel)

  def setInitial(self, ddpData, xInit, UInit):
    """ Set the initial conditions to the ddpData.
    """
    np.copyto(ddpData.intervalDataVector[0].x, xInit)
    np.copyto(ddpData.intervalDataVector[0].x_prev, xInit)
    for u, intervalData in izip(UInit,ddpData.intervalDataVector[:-1]):
      np.copyto(intervalData.u, u)
    return

  def setRunningReference(self, ddpData, Xref, name):
    index = self.costManager.getRunningCostIndex(name)
    for k, data in enumerate(ddpData.intervalDataVector[:-1]):
      cost_data = data.costData.costVector[index]
      self.costManager.runningCosts[index].setReference(cost_data, Xref[k])

  def setTerminalReference(self, ddpData, xref, name):
    index = self.costManager.getTerminalCostIndex(name)
    self.costManager.terminalCosts[index].setReference(
      ddpData.intervalDataVector[-1].costData.costVector[index], xref)