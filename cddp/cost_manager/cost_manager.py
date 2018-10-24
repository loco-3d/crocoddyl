import numpy as np

class CostManagerIntervalData(object):
  """ Calculates and stores the interval specific cost terms.
  Depends on integrator and dynamics.
  """
  def __init__(self, dynamicsModel, costsVector):
    self.costsVector = costsVector
    self.l = 0.
    self.lx = np.empty((dynamicsModel.nx(),1))
    self.lu = np.empty((dynamicsModel.nu(),1))
    self.lxx = np.empty((dynamicsModel.nx(),dynamicsModel.nx()))
    self.lux = np.empty((dynamicsModel.nu(),dynamicsModel.nx()))
    self.luu = np.empty((dynamicsModel.nu(),dynamicsModel.nu()))
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

  def createRunningData(self, ddpModel):
    return CostManagerIntervalData(ddpModel.dynamicsModel, self.runningCosts)

  def createTerminalData(self, ddpModel):
    return CostManagerIntervalData(ddpModel.dynamicsModel, self.terminalCosts)

  def addTerminal(self, cost):
    """ Add a terminal cost object to the cost manager.

    Before adding it, it checks if this is a terminal cost objects.
    """
    #assertClass(cost, 'XCost')
    self.terminalCosts.append(cost)

  def addRunning(self, cost):
    """ Add a running cost object to the cost manager.

    Before adding it, it checks if this is a terminal cost objects.
    """
    #assertClass(cost, 'XUCost')
    self.runningCosts.append(cost)

  def computeTerminalCost(self, system, data, x):
    assert self.terminalCosts > 0, "You didn't add the terminal costs"
    #assertClass(data.total, 'XCostData')

    l = data.total.l[0]
    l.fill(0.)
    for k, cost in enumerate(self.terminalCosts):
      cost_data = data.soc[k]
      l += cost.l(system, cost_data, x)
    return l

  def computeRunningCost(self, system, data, x, u, dt):
    assert self.runningCosts > 0, "You didn't add the runningCosts costs"
    #assertClass(data.total, 'XUCostData')

    # Computing the running cost
    l = data.total.l[0]
    l.fill(0.)
    for k, cost in enumerate(self.runningCosts):
      cost_data = data.soc[k]
      l += cost.l(system, cost_data, x, u)

    # Numerical integration of the cost function
    # TODO we need to use the quadrature class for this
    l *= dt
    return l

  def computeTerminalTerms(self, system, data, x):
    assert self.terminalCosts > 0, "You didn't add the terminal costs"
    #assertClass(data.total, 'XCostData')

    l = data.total.l[0]
    lx = data.total.lx
    lxx = data.total.lxx
    l.fill(0.)
    lx.fill(0.)
    lxx.fill(0.)
    for k, cost in enumerate(self.terminalCosts):
      cost_data = data.soc[k]
      l += cost.l(system, cost_data, x)
      lx += cost.lx(system, cost_data, x)
      lxx += cost.lxx(system, cost_data, x)
    return l, lx, lxx

  def computeRunningTerms(self, system, data, x, u, dt):
    assert self.runningCosts > 0, "You didn't add the running costs"
    #assertClass(data.total, 'XUCostData')

    # Computing the cost and its derivatives
    l = data.total.l[0]
    lx = data.total.lx
    lu = data.total.lu
    lxx = data.total.lxx
    luu = data.total.luu
    lux = data.total.lux
    l.fill(0.)
    lx.fill(0.)
    lu.fill(0.)
    lxx.fill(0.)
    luu.fill(0.)
    lux.fill(0.)
    for k, cost in enumerate(self.runningCosts):
      cost_data = data.soc[k]
      l += cost.l(system, cost_data, x, u)
      lx += cost.lx(system, cost_data, x, u)
      lu += cost.lu(system, cost_data, x, u)
      lxx += cost.lxx(system, cost_data, x, u)
      luu += cost.luu(system, cost_data, x, u)
      lux += cost.lux(system, cost_data, x, u)

    # Numerical integration of the cost function and its derivatives
    # TODO we need to use the quadrature class for this
    l *= dt
    lx *= dt
    lu *= dt
    lxx *= dt
    luu *= dt
    lux *= dt

    return l, lx, lu, lxx, luu, lux
