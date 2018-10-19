import numpy as np
from cost_manager_base import CostManagerBase
from cost_manager_base import CostManagerIntervalDataBase

class FloatingBaseMultibodyCostManagerIntervalData(CostManagerIntervalDataBase):
  """ Calculates and stores the interval specific cost terms.
  Depends on integrator and dynamics.
  """
  def __init__(self, costsVector):
    self.costsVector = costsVector
    self.l = 0.
    pass

  def forwardRunningCalc(self, dynamicsModel, dynamicsData):
    for cost in self.costsVector:
      cost.forwardRunningCalc(dynamicsData)
      self.l += cost.getl()

  def forwardTerminalCalc(self, dynamicsModel, dynamicsData):
    for cost in self.costsVector:
      cost.forwardTerminalCalc(dynamicsData)
      self.l += cost.getl()      
    
  
class FloatingBaseMultibodyCostManager(CostManagerBase):
  """ It computes the total cost and its derivatives for a set of running and
  terminal costs.

  The cost manager stacks a set of terminal and running cost, and from them,
  it computes the total cost and its derivatives. The derivatives are Jacobian
  and Hessian with respect to the state and control vectors. Each cost function
  and the total has its own data, which it is allocated by calling the
  createData function. Note that before doing that, you have to add the
  running and terminal cost functions of your problem.
  """

  def createRunningIntervalData(self):
    return FloatingBaseMultibodyCostManagerIntervalData(self.runningCosts)

  def createTerminalIntervalData(self):
    return FloatingBaseMultibodyCostManagerIntervalData(self.terminalCosts)

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

  def createTerminalData(self, n):
    """ Creates the data of the stack of terminal costs.

    Before creating the data, it's needed to add all the terminal costs
    of your problem.
    """
    from cddp.data import XCostData, CostManagerData
    data = CostManagerData()

    # Creating the terminal cost data, the residual data isn't needed
    data.total = XCostData(n)

    # Creating the terminal stack-of-cost data
    data.soc = [cost.createData(n) for cost in self.terminalCosts]
    return data

  def createRunningData(self, n, m):
    """ Create the data of the stack of running costs.

    Before creating the data, it's needed to add all the running costs of your
    problem.
    """
    from cddp.data import XUCostData, CostManagerData
    data = CostManagerData()

    # Creating the running cost data, the residual data isn't needed
    data.total = XUCostData(n, m)

    # Creating the running stack-of-cost data
    data.soc = [cost.createData(n, m) for cost in self.runningCosts]
    return data

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
