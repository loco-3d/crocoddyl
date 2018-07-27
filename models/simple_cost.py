import numpy as np
import cddp.quadratic_cost as qc


class GoalQuadraticCost(qc.TerminalQuadraticCost):
  """ Terminal quadratic cost
  """
  def __init__(self, goal):
    qc.TerminalQuadraticCost.__init__(self)
    self.x_des = goal

  def xr(self, data, x):
    np.copyto(data.xr, x - self.x_des)
    return data.xr

class GoalResidualQuadraticCost(qc.TerminalResidualQuadraticCost):
  """ Terminal residual-quadratic cost
  """
  def __init__(self, goal):
    k = len(goal)
    qc.TerminalResidualQuadraticCost.__init__(self, k)
    self.x_des = goal

  def r(self, data, x):
    np.copyto(data.r, x - self.x_des)
    return data.r

  def rx(self, data, x):
    np.copyto(data.rx, np.eye(data.n))
    return data.rx

class StateRunningQuadraticCost(qc.RunningQuadraticCost):
  """ State running quadratic cost
  """
  def __init__(self, goal):
    qc.RunningCost.__init__(self)
    self.x_des = goal

  def xr(self, data, x, u):
    np.copyto(data.xr, x - self.x_des)
    return data.xr

  def ur(self, data, x, u):
    # Do nothing since it's a cost in the state
    return data.ur

class StateControlRunningQuadraticCost(qc.RunningQuadraticCost):
  """ State-control running quadratic cost
  """
  def __init__(self, goal):
    qc.RunningQuadraticCost.__init__(self)
    self.x_des = goal

  def xr(self, data, x, u):
    np.copyto(data.xr, x - self.x_des)
    return data.xr

  def ur(self, data, x, u):
    np.copyto(data.ur, u)
    return data.ur