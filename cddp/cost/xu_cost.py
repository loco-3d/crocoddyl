from cddp.cost import TerminalQuadraticCost, TerminalResidualQuadraticCost
from cddp.cost import RunningQuadraticCost
import numpy as np


class StateTerminalQuadraticCost(TerminalQuadraticCost):
  """ State terminal quadratic cost
  """
  def __init__(self, goal):
    TerminalQuadraticCost.__init__(self)
    self.x_des = goal

  def xr(self, system, data, x):
    np.copyto(data.xr, system.differenceConfiguration(x, self.x_des))
    return data.xr


class StateResidualTerminalQuadraticCost(TerminalResidualQuadraticCost):
  """ Terminal residual-quadratic cost
  """
  def __init__(self, goal):
    k = len(goal)
    TerminalResidualQuadraticCost.__init__(self, k)
    self.x_des = goal

  def r(self, system, data, x):
    np.copyto(data.r, system.differenceConfiguration(x, self.x_des))
    return data.r

  def rx(self, system, data, x):
    np.copyto(data.rx, np.eye(data.n))
    return data.rx


class StateRunningQuadraticCost(RunningQuadraticCost):
  """ State running quadratic cost
  """
  def __init__(self, goal):
    RunningQuadraticCost.__init__(self)
    self.x_des = goal

  def xr(self, system, data, x, u):
    np.copyto(data.xr, system.differenceConfiguration(x, self.x_des))
    return data.xr

  def ur(self, system, data, x, u):
    # Do nothing since it's a cost in the state
    return data.ur


class StateControlRunningQuadraticCost(RunningQuadraticCost):
  """ State-control running quadratic cost
  """
  def __init__(self, goal):
    RunningQuadraticCost.__init__(self)
    self.x_des = goal

  def xr(self, system, data, x, u):
    np.copyto(data.xr, system.differenceConfiguration(x, self.x_des))
    return data.xr

  def ur(self, system, data, x, u):
    np.copyto(data.ur, u)
    return data.ur


class StateControlQuadraticRegularization(RunningQuadraticCost):
  """ State-control running quadratic cost
  """
  def __init__(self):
    RunningQuadraticCost.__init__(self)

  def xr(self, system, data, x, u):
    np.copyto(data.xr, system.differenceConfiguration(x, np.zeros_like(x)))
    return data.xr

  def ur(self, system, data, x, u):
    np.copyto(data.ur, u)
    return data.ur