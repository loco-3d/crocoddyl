import numpy as np
import cddp


class GoalQuadraticCost(cddp.TerminalQuadraticCost):
  """ Terminal quadratic cost
  """
  def __init__(self, goal):
    cddp.TerminalQuadraticCost.__init__(self)
    self.x_des = goal

  def xr(self, data, x):
    np.copyto(data.xr, x - self.x_des)
    return data.xr

class GoalResidualQuadraticCost(cddp.TerminalResidualQuadraticCost):
  """ Terminal residual-quadratic cost
  """
  def __init__(self, goal):
    k = len(goal)
    cddp.TerminalResidualQuadraticCost.__init__(self, k)
    self.x_des = goal

  def r(self, data, x):
    np.copyto(data.r, x - self.x_des)
    return data.r

  def rx(self, data, x):
    np.copyto(data.rx, np.eye(data.n))
    return data.rx

class StateRunningQuadraticCost(cddp.RunningQuadraticCost):
  """ State running quadratic cost
  """
  def __init__(self, goal):
    cddp.RunningCost.__init__(self)
    self.x_des = goal

  def xr(self, data, x, u):
    np.copyto(data.xr, x - self.x_des)
    return data.xr

  def ur(self, data, x, u):
    # Do nothing since it's a cost in the state
    return data.ur

class StateControlRunningQuadraticCost(cddp.RunningQuadraticCost):
  """ State-control running quadratic cost
  """
  def __init__(self, goal):
    cddp.RunningQuadraticCost.__init__(self)
    self.x_des = goal

  def xr(self, data, x, u):
    np.copyto(data.xr, x - self.x_des)
    return data.xr

  def ur(self, data, x, u):
    np.copyto(data.ur, u)
    return data.ur

class StateControlQuadraticRegularization(cddp.RunningQuadraticCost):
  """ State-control running quadratic cost
  """
  def __init__(self):
    cddp.RunningQuadraticCost.__init__(self)

  def xr(self, data, x, u):
    np.copyto(data.xr, x)
    return data.xr

  def ur(self, data, x, u):
    np.copyto(data.ur, u)
    return data.ur