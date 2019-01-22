from crocoddyl.costs.cost import TerminalCostData, RunningCostData
import numpy as np


class TerminalCostManagerData(TerminalCostData):
  def __init__(self, dynamicModel, costVector):
    # Including the data structure of the terminal cost
    TerminalCostData.__init__(self, dynamicModel)

    # Including the data structure for each individual terminal cost functions
    self.costVector = []
    for cost in costVector:
      self.costVector.append(cost.createData(dynamicModel))


class RunningCostManagerData(RunningCostData):
  def __init__(self, dynamicModel, costVector):
    # Including the data structure of the running cost
    RunningCostData.__init__(self, dynamicModel)

    # Including the data structure for each individual terminal cost functions
    self.costVector = []
    for cost in costVector:
      self.costVector.append(cost.createData(dynamicModel))


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

  def createTerminalData(self, dynamicModel):
    """ Create the terminal cost data for a given dynamic model.

    :param dynamicModel: dynamic model
    """
    return TerminalCostManagerData(dynamicModel, self.terminalCosts)

  def createRunningData(self, dynamicModel):
    """ Create the running cost data for a given dynamic model.

    :param dynamicModel: dynamic model
    """
    return RunningCostManagerData(dynamicModel, self.runningCosts)

  def addTerminal(self, cost, name):
    """ Add a terminal cost function.

    :param cost: cost function
    :param name: cost function name
    """
    index = len(self.terminalCosts)
    self.terminalCostDict[name] = index
    self.terminalCosts.append(cost)

  def addRunning(self, cost, name):
    """ Add a running cost function.

    :param cost: cost function
    :param name: cost function name
    """
    index = len(self.runningCosts)
    self.runningCostDict[name] = index
    self.runningCosts.append(cost)

  def removeTerminal(self, name):
    """ Remove a terminal cost function given its name.

    :param name: cost function name
    """
    if name in self.terminalCostDict:
      index = self.getTerminalCostIndex(name)
      del self.terminalCostDict[name]
      del self.terminalCosts[index]

  def removeRunning(self, name):
    """ Remove a running cost function given its name.

    :param name: cost function name
    """
    if name in self.runningCostDict:
      index = self.getRunningCostIndex(name)
      del self.runningCostDict[name]
      del self.runningCosts[index]

  # Static functions that defines all cost computations
  def forwardRunningCalc(costManager, costData, dynamicData, x, u):
    costData.l = 0.
    for k, cost in enumerate(costManager.runningCosts):
      cost.updateCost(costData.costVector[k], dynamicData, x, u)
      costData.l += cost.getl(costData.costVector[k])

  def forwardTerminalCalc(costManager, costData, dynamicData, x):
    costData.l = 0.
    for k, cost in enumerate(costManager.terminalCosts):
      cost.updateCost(costData.costVector[k], dynamicData, x, u = None)
      costData.l += cost.getl(costData.costVector[k])

  def backwardRunningCalc(costManager, costData, dynamicData, x, u):
    for k, cost in enumerate(costManager.runningCosts):
      cost.updateQuadraticAppr(costData.costVector[k], dynamicData, x, u)

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

  def backwardTerminalCalc(costManager, costData, dynamicData, x):
    for k, cost in enumerate(costManager.terminalCosts):
      cost.updateQuadraticAppr(costData.costVector[k], dynamicData, x, u = None)

    costData.lx.fill(0.)
    costData.lxx.fill(0.)
    for k, cost in enumerate(costManager.terminalCosts):
      costData.lx += cost.getlx(costData.costVector[k])
      costData.lxx += cost.getlxx(costData.costVector[k])

  def getTerminalCost(self, name):
    """ Return the running cost model given its name.

    :param name: cost function name
    """
    return self.terminalCosts[self.getTerminalCostIndex(name)]

  def getRunningCost(self, name):
    """ Return the running cost model given its name.

    :param name: cost function name
    """
    return self.runningCosts[self.getRunningCostIndex(name)]

  def getTerminalCostIndex(self, name):
    """ Get the index of a given terminal cost name.

    :param name: cost function name
    """
    # Returning the index of the cost
    return self.terminalCostDict[name]

  def getRunningCostIndex(self, name):
    """ Get the index of a given terminal cost name.

    :param name: cost function name
    """
    # Returning the index of the cost
    return self.runningCostDict[name]