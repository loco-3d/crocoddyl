import numpy as np

class CostManagerIntervalData(object):
  """ Calculates and stores the interval specific cost terms.
  Depends on integrator and dynamics.
  """
  def __init__(self, dynamicsModel, costsVector):
    self.costsVector = costsVector
    self.l = 0.
    self.lx = np.zeros((dynamicsModel.nx(),1))
    self.lu = np.zeros((dynamicsModel.nu(),1))
    self.lxx = np.zeros((dynamicsModel.nx(),dynamicsModel.nx()))
    self.lux = np.zeros((dynamicsModel.nu(),dynamicsModel.nx()))
    self.luu = np.zeros((dynamicsModel.nu(),dynamicsModel.nu()))
    return

  def forwardRunningCalc(self, dynamicsModel, dynamicsData):
    self.l = 0.
    for cost in self.costsVector:
      cost.forwardRunningCalc(dynamicsData)
      self.l += cost.getl()

  def forwardTerminalCalc(self, dynamicsModel, dynamicsData):
    self.l = 0.
    for cost in self.costsVector:
      cost.forwardTerminalCalc(dynamicsData)
      self.l += cost.getl()
    #TODO: THIS IS STUPID!!!
    self.l *=1000.

  def backwardRunningCalc(self, dynamicsModel, dynamicsData):
    for cost in self.costsVector:
      cost.backwardRunningCalc(dynamicsData)
    self.lx.fill(0.)
    self.lu.fill(0.)
    self.lxx.fill(0.)
    self.lux.fill(0.)
    self.luu.fill(0.)

    for cost in self.costsVector:
      self.lx += cost.getlx()
      self.lu += cost.getlu()
      self.lxx += cost.getlxx()
      self.lux += cost.getlux()
      self.luu += cost.getluu()
    return

  def backwardTerminalCalc(self,dynamicsModel, dynamicsData):
    for cost in self.costsVector:
      cost.backwardTerminalCalc(dynamicsData)
    self.lx.fill(0.)
    self.lxx.fill(0.)
    for cost in self.costsVector:
      self.lx += cost.getlx()
      self.lxx += cost.getlxx()
    #TODO: THIS IS STUPID!!!
    self.lx *= 1000.
    self.lxx *= 1000.
    return

      
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
    self.runningCosts = []
    self.terminalCosts = []
    pass
  
  def createRunningData(self, ddpModel):
    return CostManagerIntervalData(ddpModel.dynamicsModel, self.runningCosts)

  def createTerminalData(self, ddpModel):
    return CostManagerIntervalData(ddpModel.dynamicsModel, self.terminalCosts)

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
