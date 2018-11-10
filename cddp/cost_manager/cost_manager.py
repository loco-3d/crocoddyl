import numpy as np


class TerminalCostData(object):
  """ Data structure for the terminal cost of a specific time interval.

  The terminal cost terms are the cost value and its derivates with respect to
  the state. The dimensions of these vector or matrices depends on the dynamic
  model.
  """
  def __init__(self, dynamicsModel):
    """ Creates the terminal cost data.

    :param dynamicsModel: dynanic model
    """
    self.l = 0.
    self.lx = np.zeros((dynamicsModel.nx(),1))
    self.lxx = np.zeros((dynamicsModel.nx(),dynamicsModel.nx()))


class RunningCostData(TerminalCostData):
  """ Data structure for the running cost of a specific time interval.

  The running cost terms includes the terminal ones plus its derivates
  with respect to the control. The dimensions of these vector or matrices
  depends on the dynamic
  model.
  """
  def __init__(self, dynamicsModel):
    """ Creates the running cost data.

    :param dynamicsModel: dynanic model
    """
    TerminalCostData.__init__(self, dynamicsModel)
    self.lu = np.zeros((dynamicsModel.nu(),1))
    self.lux = np.zeros((dynamicsModel.nu(),dynamicsModel.nx()))
    self.luu = np.zeros((dynamicsModel.nu(),dynamicsModel.nu()))


class CostManager(object):
  """ Stacks a set of terminal and running cost functions.

  It stacks a set of terminal and running cost used for computing the cost
  values and its derivatives at each time interval. Both terminal and running
  costs requires its own data, which it is created by calling the
  createTerminalData or createRunningData functions. Note that before doing
  that, you have to add the running and terminal cost functions of your problem.
  Static functions define the routines used for computing the cost value and
  its derivatives.
  """
  def __init__(self):
    """ Construct the internal vector of terminal and cost functions.
    """
    self.terminalCosts = []
    self.runningCosts = []

  @staticmethod
  def createTerminalData(dynamicsModel):
    """ Creates the terminal cost data for a given dynamic model

    :param dynamicsModel: dynamic model
    """
    return TerminalCostData(dynamicsModel)

  @staticmethod
  def createRunningData(dynamicsModel):
    """ Creates the running cost data for a given dynamic model

    :param dynamicsModel: dynamic model
    """
    return RunningCostData(dynamicsModel)

  def addTerminal(self, cost):
    """ Adds a terminal cost function to the cost model.
    Before adding it, it checks if this is a terminal cost objects.
    """
    self.terminalCosts.append(cost)

  def addRunning(self, cost):
    """ Adds a running cost function to the cost model.
    Before adding it, it checks if this is a terminal cost objects.
    """
    self.runningCosts.append(cost)

  # Static functions that defines all cost computations
  def forwardRunningCalc(costManager, costData, dynamicsData):
    costData.l = 0.
    for cost in costManager.runningCosts:
      cost.updateCost(dynamicsData)
      costData.l += cost.getl()

  def forwardTerminalCalc(costManager, costData, dynamicsData):
    costData.l = 0.
    for cost in costManager.terminalCosts:
      cost.updateCost(dynamicsData)
      costData.l += cost.getl()

  def backwardRunningCalc(costManager, costData, dynamicsData):
    for cost in costManager.runningCosts:
      cost.updateQuadraticAppr(dynamicsData)

    costData.lx.fill(0.)
    costData.lu.fill(0.)
    costData.lxx.fill(0.)
    costData.lux.fill(0.)
    costData.luu.fill(0.)
    for cost in costManager.runningCosts:
      costData.lx += cost.getlx()
      costData.lu += cost.getlu()
      costData.lxx += cost.getlxx()
      costData.lux += cost.getlux()
      costData.luu += cost.getluu()

  def backwardTerminalCalc(costManager, costData, dynamicsData):
    for cost in costManager.terminalCosts:
      cost.updateQuadraticAppr(dynamicsData)

    costData.lx.fill(0.)
    costData.lxx.fill(0.)
    for cost in costManager.terminalCosts:
      costData.lx += cost.getlx()
      costData.lxx += cost.getlxx()
