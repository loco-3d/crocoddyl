from cddp.costs.cost import TerminalCostData, RunningCostData
import numpy as np


class TerminalCostManagerData(TerminalCostData):
  def __init__(self, dynamicsModel, costVector):
    # Including the data structure of the terminal cost
    TerminalCostData.__init__(self, dynamicsModel)

    # Including the data structure for each individual terminal cost functions
    self.costVector = []
    for cost in costVector:
      self.costVector.append(cost.createData(dynamicsModel))


class RunningCostManagerData(RunningCostData):
  def __init__(self, dynamicsModel, costVector):
    # Including the data structure of the running cost
    RunningCostData.__init__(self, dynamicsModel)

    # Including the data structure for each individual terminal cost functions
    self.costVector = []
    for cost in costVector:
      self.costVector.append(cost.createData(dynamicsModel))


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
    self.terminalCostDict = {}
    self.runningCostDict = {}

  def createTerminalData(self, dynamicsModel):
    """ Creates the terminal cost data for a given state dimension

    :param dynamicsModel: dynamics model
    """
    return TerminalCostManagerData(dynamicsModel, self.terminalCosts)

  def createRunningData(self, dynamicsModel):
    """ Creates the running cost data for a given state and control dimension

    :param dynamicsModel: dynamics model
    """
    return RunningCostManagerData(dynamicsModel, self.runningCosts)

  def addTerminal(self, cost, name):
    """ Adds a terminal cost function to the cost model.
    Before adding it, it checks if this is a terminal cost objects.
    """
    index = len(self.terminalCosts)
    self.terminalCostDict[name] = index
    self.terminalCosts.append(cost)

  def addRunning(self, cost, name):
    """ Adds a running cost function to the cost model.
    Before adding it, it checks if this is a terminal cost objects.
    """
    index = len(self.runningCosts)
    self.runningCostDict[name] = index
    self.runningCosts.append(cost)

  # Static functions that defines all cost computations
  def forwardRunningCalc(costManager, costData, dynamicsData):
    costData.l = 0.
    for k, cost in enumerate(costManager.runningCosts):
      cost.updateCost(costData.costVector[k], dynamicsData)
      costData.l += cost.getl(costData.costVector[k])

  def forwardTerminalCalc(costManager, costData, dynamicsData):
    costData.l = 0.
    for k, cost in enumerate(costManager.terminalCosts):
      cost.updateCost(costData.costVector[k], dynamicsData)
      costData.l += cost.getl(costData.costVector[k])

  def backwardRunningCalc(costManager, costData, dynamicsData):
    for k, cost in enumerate(costManager.runningCosts):
      cost.updateQuadraticAppr(costData.costVector[k], dynamicsData)

    costData.lx.fill(0.)
    costData.lu.fill(0.)
    costData.lxx.fill(0.)
    costData.lux.fill(0.)
    costData.luu.fill(0.)
    for k, cost in enumerate(costManager.runningCosts):
      costData.lx += cost.getlx(costData.costVector[k])
      costData.lu += cost.getlu(costData.costVector[k])
      costData.lxx += cost.getlxx(costData.costVector[k])
      costData.lux += cost.getlux(costData.costVector[k])
      costData.luu += cost.getluu(costData.costVector[k])

  def backwardTerminalCalc(costManager, costData, dynamicsData):
    for k, cost in enumerate(costManager.terminalCosts):
      cost.updateQuadraticAppr(costData.costVector[k], dynamicsData)

    costData.lx.fill(0.)
    costData.lxx.fill(0.)
    for k, cost in enumerate(costManager.terminalCosts):
      costData.lx += cost.getlx(costData.costVector[k])
      costData.lxx += cost.getlxx(costData.costVector[k])

  def getTerminalCost(self, name):
    """ Return the running cost model given its name

    :param name: cost function name
    """
    return self.terminalCosts[self.getTerminalCostIndex(name)]

  def getRunningCost(self, name):
    """ Return the running cost model given its name

    :param name: cost function name
    """
    return self.runningCosts[self.getRunningCostIndex(name)]

  def getTerminalCostIndex(self, name):
    """ Get the index of a given terminal cost name

    :param name: cost function name
    """
    # Returning the index of the cost
    return self.terminalCostDict[name]

  def getRunningCostIndex(self, name):
    """ Get the index of a given terminal cost name

    :param name: cost function name
    """
    # Returning the index of the cost
    return self.runningCostDict[name]