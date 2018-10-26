import numpy as np

class CostManagerIntervalData(object):
  """ Calculates and stores the interval specific cost terms.
  Depends on integrator and dynamics.
  """
  def __init__(self, dynamicsModel, costsVector):
    self.costsVector = costsVector


class CostManagerTerminalData(CostManagerIntervalData):
  def __init__(self, dynamicsModel, costsVector):
    CostManagerIntervalData.__init__(self, dynamicsModel, costsVector)
    self.l = 0.
    self.lx = np.zeros((dynamicsModel.nx(),1))
    self.lxx = np.zeros((dynamicsModel.nx(),dynamicsModel.nx()))


class CostManagerRunningData(CostManagerTerminalData):
  def __init__(self, dynamicsModel, costsVector):
    CostManagerTerminalData.__init__(self, dynamicsModel, costsVector)
    self.lu = np.zeros((dynamicsModel.nu(),1))
    self.lux = np.zeros((dynamicsModel.nu(),dynamicsModel.nx()))
    self.luu = np.zeros((dynamicsModel.nu(),dynamicsModel.nu()))


class CostManager(object):
  """ It computes the total cost and its derivatives for a set of running and
  terminal costs.

  The cost manager stacks a set of terminal and running cost, and from them,
  it computes the total cost and its derivatives. The derivatives are Jacobian
  and Hessian with respect to the state and control vectors. Each cost function
  and the total has its own data, which it is allocated by calling the
  createData function. Note that before doing that, you have to add the
  running and terminal cost functions of your problem.
  """

  def __init__(self,dynamicsModel):
    self.dynamicsModel = dynamicsModel
    self.terminalCosts = []
    self.runningCosts = []

  def createTerminalData(self, ddpModel):
    return CostManagerTerminalData(ddpModel.dynamicsModel, self.terminalCosts)

  def createRunningData(self, ddpModel):
    return CostManagerRunningData(ddpModel.dynamicsModel, self.runningCosts)

  def addTerminal(self, cost):
    """ Add a terminal cost object to the cost manager.
    Before adding it, it checks if this is a terminal cost objects.
    """
    self.terminalCosts.append(cost)

  def addRunning(self, cost):
    """ Add a running cost object to the cost manager.

    Before adding it, it checks if this is a terminal cost objects.
    """
    self.runningCosts.append(cost)

  def forwardRunningCalc(self, costData, dynamicsData):
    costData.l = 0.
    for cost in costData.costsVector:
      cost.forwardRunningCalc(dynamicsData)
      costData.l += cost.getl()

  def forwardTerminalCalc(self, costData, dynamicsData):
    costData.l = 0.
    for cost in costData.costsVector:
      cost.forwardTerminalCalc(dynamicsData)
      costData.l += cost.getl()
    #TODO: THIS IS STUPID!!!
    costData.l *=1000.

  def backwardRunningCalc(self, costData, dynamicsData):
    for cost in costData.costsVector:
      cost.backwardRunningCalc(dynamicsData)

    costData.lx.fill(0.)
    costData.lu.fill(0.)
    costData.lxx.fill(0.)
    costData.lux.fill(0.)
    costData.luu.fill(0.)
    for cost in costData.costsVector:
      costData.lx += cost.getlx()
      costData.lu += cost.getlu()
      costData.lxx += cost.getlxx()
      costData.lux += cost.getlux()
      costData.luu += cost.getluu()
    return

  def backwardTerminalCalc(self, costData, dynamicsData):
    for cost in costData.costsVector:
      cost.backwardTerminalCalc(dynamicsData)

    costData.lx.fill(0.)
    costData.lxx.fill(0.)
    for cost in costData.costsVector:
      costData.lx += cost.getlx()
      costData.lxx += cost.getlxx()
    #TODO: THIS IS STUPID!!!
    costData.lx *= 1000.
    costData.lxx *= 1000.
    return