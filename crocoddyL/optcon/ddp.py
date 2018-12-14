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
    self.x = np.zeros((ddpModel.dynamicModel.nxImpl(), 1))
    self.x_prev = np.zeros((ddpModel.dynamicModel.nxImpl(), 1))

    # Dynamic and cost data of the terminal interval
    self.dynamic = ddpModel.createTerminalDynamicData(tFinal)
    self.cost = ddpModel.createTerminalCostData()

    # Value function derivatives of the terminal interval
    self.Vx = np.zeros((ddpModel.dynamicModel.nx(), 1))
    self.Vxx = np.zeros((ddpModel.dynamicModel.nx(),ddpModel.dynamicModel.nx()))


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
    self.x = np.zeros((ddpModel.dynamicModel.nxImpl(), 1))
    self.u = np.zeros((ddpModel.dynamicModel.nu(), 1))
    self.x_prev = np.zeros((ddpModel.dynamicModel.nxImpl(), 1))
    self.u_prev = np.zeros((ddpModel.dynamicModel.nu(), 1))

    # Dynamic and cost data of the running interval
    self.dynamic = ddpModel.createRunningDynamicData(tInit, tFinal - tInit)
    self.cost = ddpModel.createRunningCostData()

    # Value function derivatives of the running interval
    self.Vx = np.zeros((ddpModel.dynamicModel.nx(), 1))
    self.Vxx = np.zeros((ddpModel.dynamicModel.nx(),ddpModel.dynamicModel.nx()))

    # Quadratic approximation of the Value function
    self.Qx = np.zeros((ddpModel.dynamicModel.nx(), 1))
    self.Qu = np.zeros((ddpModel.dynamicModel.nu(), 1))
    self.Qxx = np.zeros((ddpModel.dynamicModel.nx(),ddpModel.dynamicModel.nx()))
    self.Qux = np.zeros((ddpModel.dynamicModel.nu(),ddpModel.dynamicModel.nx()))
    self.Quu = np.zeros((ddpModel.dynamicModel.nu(),ddpModel.dynamicModel.nu()))
    self.Quu_r = np.zeros((ddpModel.dynamicModel.nu(),ddpModel.dynamicModel.nu()))
    self.Qux_r = np.zeros((ddpModel.dynamicModel.nu(),ddpModel.dynamicModel.nx()))
    self.L = np.zeros((ddpModel.dynamicModel.nu(),ddpModel.dynamicModel.nu()))
    self.L_inv = np.zeros((ddpModel.dynamicModel.nu(),ddpModel.dynamicModel.nu()))
    self.Quu_inv_minus = np.zeros((ddpModel.dynamicModel.nu(),ddpModel.dynamicModel.nu()))

    # Feedback and feed-forward terms
    self.K = np.zeros((ddpModel.dynamicModel.nu(), ddpModel.dynamicModel.nx()))
    self.j = np.zeros((ddpModel.dynamicModel.nu(), 1))

    # Extra DDP terms
    self.jt_Quu_j = 0.
    self.jt_Qu = 0.
    self.Vpxx_fx = np.zeros((ddpModel.dynamicModel.nx(), ddpModel.dynamicModel.nx()))
    self.Vpxx_fu = np.zeros((ddpModel.dynamicModel.nx(), ddpModel.dynamicModel.nu()))


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
    self.interval = [DDPRunningIntervalData(ddpModel, timeline[i], timeline[i+1])
                     for i in xrange(self.N)]
    self.interval.append(DDPTerminalIntervalData(ddpModel, timeline[-1]))

    # Cost-related data
    self.totalCost = 0.
    self.totalCost_prev = 0.
    self.dV_exp = 0.
    self.dV = 0.
    self.total_jt_Quu_j = 0.
    self.total_jt_Qu = 0.

    # Line search and regularization data
    self.alpha = -1.
    self.muLM = -1.
    self.muV = -1.
    self.muI = np.zeros((ddpModel.dynamicModel.nu(),ddpModel.dynamicModel.nu()))

    # Convergence and stopping criteria data
    self._convergence = False
    self.backward_status = " "
    self.forward_status = " "
    self.gamma = 0.
    self.theta = 0.
    self.z_new = 0.
    self.z = 0.


class DDPModel(object):
  """ Class to save the model information for the system, cost and dynamics
  """
  def __init__(self, dynamicModel, costManager):
    self.dynamicModel = dynamicModel
    self.costManager = costManager

  def forwardTerminalCalc(self, dynamicData, costData, x):
    """Performes the dynamics integration to generate the state and control functions"""
    self.dynamicModel.forwardTerminalCalc(dynamicData, x)
    self.costManager.forwardTerminalCalc(costData, dynamicData, x)

  def forwardRunningCalc(self, dynamicData, costData, x, u, xNext):
    """Performes the dynamics integration to generate the state and control functions"""
    self.dynamicModel.forwardRunningCalc(dynamicData, x, u, xNext)
    self.costManager.forwardRunningCalc(costData, dynamicData, x, u)

  def backwardTerminalCalc(self, dynamicData, costData, x):
    """Performs the calculations before the backward pass
    Pinocchio Data has already been filled with the forward pass."""
    self.dynamicModel.backwardTerminalCalc(dynamicData, x)
    self.costManager.backwardTerminalCalc(costData, dynamicData, x)

  def backwardRunningCalc(self, dynamicData, costData, x, u):
    """Performs the calculations before the backward pass"""
    self.dynamicModel.backwardRunningCalc(dynamicData, x, u)
    self.costManager.backwardRunningCalc(costData, dynamicData, x, u)

  def createRunningDynamicData(self, tInit, dt):
    return self.dynamicModel.createData(tInit, dt)

  def createRunningCostData(self):
    return self.costManager.createRunningData(self.dynamicModel)

  def createTerminalDynamicData(self, tFinal):
    return self.dynamicModel.createData(tFinal, 0.)

  def createTerminalCostData(self):
    return self.costManager.createTerminalData(self.dynamicModel)

  def setInitial(self, ddpData, xInit, UInit):
    """ Set the initial conditions to the ddpData.
    """
    np.copyto(ddpData.interval[0].x, xInit)
    np.copyto(ddpData.interval[0].x_prev, xInit)
    for u, intervalData in izip(UInit,ddpData.interval[:-1]):
      np.copyto(intervalData.u, u)
    return

  def setRunningReference(self, ddpData, Xref, name):
    index = self.costManager.getRunningCostIndex(name)
    for k, data in enumerate(ddpData.interval[:-1]):
      cost_data = data.cost.costVector[index]
      self.costManager.runningCosts[index].setReference(cost_data, Xref[k])

  def setTerminalReference(self, ddpData, xref, name):
    index = self.costManager.getTerminalCostIndex(name)
    self.costManager.terminalCosts[index].setReference(
      ddpData.interval[-1].cost.costVector[index], xref)