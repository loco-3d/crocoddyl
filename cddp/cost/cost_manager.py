import abc
import numpy as np
from cddp.utils import assertClass


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
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    self.terminal = []
    self.running = []

  def addTerminal(self, cost):
    """ Add a terminal cost object to the cost manager.

    Before adding it, it checks if this is a terminal cost objects.
    """
    assertClass(cost, 'XCost')
    self.terminal.append(cost)

  def addRunning(self, cost):
    """ Add a running cost object to the cost manager.

    Before adding it, it checks if this is a terminal cost objects.
    """
    assertClass(cost, 'XUCost')
    self.running.append(cost)

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
    data.soc = [cost.createData(n) for cost in self.terminal]
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
    data.soc = [cost.createData(n, m) for cost in self.running]
    return data

  def computeTerminalCost(self, data, x):
    assert self.terminal > 0, "You didn't add the terminal costs"
    assertClass(data.total, 'XCostData')

    l = data.total.l[0]
    l.fill(0.)
    for k, cost in enumerate(self.terminal):
      cost_data = data.soc[k]
      l += cost.l(cost_data, x)
    return l

  def computeRunningCost(self, data, x, u):
    assert self.running > 0, "You didn't add the running costs"
    assertClass(data.total, 'XUCostData')
    
    l = data.total.l[0]
    l.fill(0.)
    for k, cost in enumerate(self.running):
      cost_data = data.soc[k]
      l += cost.l(cost_data, x, u)
    return l

  def computeTerminalTerms(self, data, x):
    assert self.terminal > 0, "You didn't add the terminal costs"
    assertClass(data.total, 'XCostData')
    
    l = data.total.l[0]
    lx = data.total.lx
    lxx = data.total.lxx
    l.fill(0.)
    lx.fill(0.)
    lxx.fill(0.)
    for k, cost in enumerate(self.terminal):
      cost_data = data.soc[k]
      l += cost.l(cost_data, x)
      lx += cost.lx(cost_data, x)
      lxx += cost.lxx(cost_data, x)
    return l, lx, lxx

  def computeRunningTerms(self, data, x, u):
    assert self.running > 0, "You didn't add the running costs"
    assertClass(data.total, 'XUCostData')
    
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
    for k, cost in enumerate(self.running):
      cost_data = data.soc[k]
      l += cost.l(cost_data, x, u)
      lx += cost.lx(cost_data, x, u)
      lu += cost.lu(cost_data, x, u)
      lxx += cost.lxx(cost_data, x, u)
      luu += cost.luu(cost_data, x, u)
      lux += cost.lux(cost_data, x, u)
    return l, lx, lu, lxx, luu, lux

  # def lx(self, data, x):
  #     lx = data.total.lx
  #     lx.fill(0.)

  #     for k, cost in enumerate(self.terminal):
  #             cost_data = data.terminal[k]
  #         cost_value += cost.l(cost_data, x)
  #     data.total.l[0] = cost_value
  #     return cost_value

  # @abc.abstractmethod
  # def computeFinalCost(self, t, x): pass

  # @abc.abstractmethod
  # def computeAllTerms(self, t, x, u): pass

  # @abc.abstractmethod
  # def computeFinalTerms(self, t, x): pass
