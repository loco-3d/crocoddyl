import abc
from cddp.cost import RunningResidualCost
from cddp.utils import assertGreaterThan
import numpy as np


class XLogBarrier(RunningResidualCost):
  """ Log barrier function for approximating state inequality constraints.

  A log-barrier function replaces an inequality constraint by penalizing term
  in the cost function. The barrier function is an approximation of
  an inequality constraint r(x) < b, where b are the bounds and r() the
  constraint function. The amount of smoothing can be controlled through a
  free parameter \mu > 0. Note that in the limit \mu -> 0 resembles the exact
  inequality constraint. Formally, the log-barrier function \phi is defined as:
  \mu * sum_i [-log(b_i-r_i(x))] for r(x) < b and infinity otherwise.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, bound):
    self._bound = bound
    self._mu = 0.1
    self._inf = 10e5
    RunningResidualCost.__init__(self, self._bound.size)

  def l(self, data, x, u):
    # Compute the inequality constraint function
    r = self.r(data, x, u)

    # Compute the log-barrier function and store the result in data
    data.l[0] = 0.
    for i in range(data.k):
      g = self._bound[i] - r[i]
      if g > 0:
        data.l[0] += -self._mu * np.log(g)
      else:
        data.l[0] += self._inf
        return data.l[0] # it doesn't make sense to continue
    return data.l[0]

  def lx(self, data, x, u):
    # Compute the inequality constraint function and its state Jacobian. Note
    # that for efficiency we assume that you run l() first
    r = data.r
    rx = self.rx(data, x, u)

    # Compute the state Jacobian of the log-barrier function and store the
    # result in data
    data.lx *= 0.
    for i in range(data.k):
      g = self._bound[i] - r[i]
      if g > 0:
        data.lx += self._mu * rx[i,:].T / g
      else:
        data.lx += rx[i,:].T * self._inf
        return data.lx # it doesn't make sense to continue
    return data.lx

  def lxx(self, data, x, u):
    # Compute the inequality constraint function and its state Jacobian. We
    # neglect the Hessian of the inequality constraint. Additionally, for
    # efficiency we assume that you run l() and lx() first
    r = data.r
    rx = data.rx

    # Compute the state Hessian of the log-barrier function and store the
    # result in data
    data.lxx *= 0.
    for i in range(data.k):
      g = self._bound[i] - r[i]
      if g > 0:
        data.lxx += self._mu * rx[i,:].T * rx[i,:] / (g * g)
      else:
        data.lxx += rx[i,:].T * rx[i,:] * self._inf
        return data.lxx  # it doesn't make sense to continue
    return data.lxx

  def ru(self, data, x, u):
    # This residual vector is zero, so we do anything.
    return data.ru

  def lu(self, data, x, u):
    # This derivative is zero, so we do anything.
    return data.lu

  def luu(self, data, x, u):
    # This derivative is zero, so we do anything.
    return data.luu

  def lux(self, data, x, u):
    # This derivative is zero, so we do anything.
    return data.lux


class ULogBarrier(RunningResidualCost):
  """ Log barrier function for approximating inequality constraints.

  A log-barrier function replaces an inequality constraint by penalizing term
  in the cost function. The barrier function is an approximation of
  an inequality constraint r(u) < b, where b are the bounds and r() the
  constraint function. The amount of smoothing can be controlled through a
  free parameter \mu > 0. Note that in the limit \mu -> 0 resembles the exact
  inequality constraint. Formally, the log-barrier function \phi is defined as:
  \mu * sum_i [-log(b_i-r_i(u))] for r(u) < b and infinity otherwise.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, bound):
    self._bound = bound
    self._mu = 1.
    self._inf = np.finfo(float).max
    RunningResidualCost.__init__(self, self._bound.size)

  def l(self, data, x, u):
    # Compute the inequality constraint function
    r = self.r(data, x, u)

    # Compute the log-barrier function and store the result in data
    data.l[0] = 0.
    for i in range(data.k):
      g = self._bound[i] - r[i]
      if g > 0:
        data.l[0] += -self._mu * np.log(g)
      else:
        data.l[0] += self._inf
        return data.l[0] # it doesn't make sense to continue
    return data.l[0]

  def lu(self, data, x, u):
    # Compute the inequality constraint function and its control Jacobian. Note
    # that for efficiency we assume that you run l() first
    r = data.r
    ru = self.ru(data, x, u)

    # Compute the control Jacobian of the log-barrier function and store the
    # result in data
    data.lu *= 0.
    for i in range(data.k):
      g = self._bound[i] - r[i]
      if g > 0:
        data.lu += self._mu * ru[i,:].T / g
      else:
        data.lu += ru[i,:].T * self._inf
        return data.lu  # it doesn't make sense to continue
    return data.lu

  def luu(self, data, x, u):
    # Compute the inequality constraint function and its control Jacobian. We
    # neglect the Hessian of the inequality constraint. Additionally, for
    # efficiency we assume that you run l() and lu() first
    r = data.r
    ru = data.ru

    # Compute the control Hessian of the log-barrier function and store the
    # result in data
    data.luu *= 0.
    for i in range(data.k):
      g = self._bound[i] - r[i]
      if g > 0:
        data.luu += self._mu * ru[i,:].T * ru[i,:] / (g * g)
      else:
        data.luu += ru[i,:].T * ru[i,:] * self._inf
        return data.luu  # it doesn't make sense to continue
    return data.luu

  def rx(self, data, x, u):
    # This residual vector is zero, so we do anything.
    return data.rx

  def lx(self, data, x, u):
    # This residual vector is zero, so we do anything.
    return data.lx
  
  def lxx(self, data, x, u):
    # This residual vector is zero, so we do anything.
    return data.lxx

  def lux(self, data, x, u):
    # This residual vector is zero, so we do anything.
    return data.lux


class XULogBarrier(XLogBarrier):
  """ Log barrier function for approximating inequality constraints.

  A log-barrier function replaces an inequality constraint by penalizing term
  in the cost function. The barrier function is an approximation of
  an inequality constraint r(x,u) < b, where b are the bounds and r() the
  constraint function. The amount of smoothing can be controlled through a
  free parameter \mu > 0. Note that in the limit \mu -> 0 resembles the exact
  inequality constraint. Formally, the log-barrier function \phi is defined as:
  \mu * sum_i [-log(b_i-r_i(x,u))] for r(x,u) < b and infinity otherwise.

  An import remark here is that for inequality constraints that only depends
  on the state (or control) is more efficient to use XLogBarrier (or
  ULogBarrier) class since we avoid unnecessary computation.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, bound):
    self._bound = bound
    self._mu = 1.
    self._inf = np.finfo(float).max
    XLogBarrier.__init__(self, self._bound.size)

  @abc.abstractmethod
  def ru(self, data, x, u): pass

  def lu(self, data, x, u):
    # Compute the inequality constraint function and its control Jacobian. Note
    # that for efficiency we assume that you run l() first
    r = data.r
    ru = self.ru(data, x, u)

    # Compute the control Jacobian of the log-barrier function and store the
    # result in data
    data.lu *= 0.
    for i in range(data.k):
      g = self._bound[i] - r[i]
      if g > 0:
        data.lu += self._mu * ru[i,:].T / g
      else:
        data.lu += ru[i,:].T * self._inf
        return data.lu  # it doesn't make sense to continue
    return data.lu

  def luu(self, data, x, u):
    # Compute the inequality constraint function and its control Jacobian. We
    # neglect the Hessian of the inequality constraint. Additionally, for
    # efficiency we assume that you run l() and lu() first
    r = data.r
    ru = data.ru

    # Compute the control Hessian of the log-barrier function and store the
    # result in data
    data.luu *= 0.
    for i in range(data.k):
      g = self._bound[i] - r[i]
      if g > 0:
        data.luu += self._mu * ru[i,:].T * ru[i,:] / (g * g)
      else:
        data.luu += ru[i,:].T * ru[i,:] * self._inf
        return data.luu  # it doesn't make sense to continue
    return data.luu

  def lux(self, data, x, u):
    # Compute the inequality constraint function and its state and control
    # Jacobian. We neglect the Hessians of the inequality constraint.
    # Additionally, for efficiency we assume that you run l(), lx() and lu()
    # first
    r = data.r
    rx = data.rx
    ru = data.ru
    data.lux *= 0.
    for i in range(data.k):
      g = self._bound[i] - r[i]
      if g > 0:
        data.lux += self._mu * ru[i,:].T * rx[i,:] / (g * g)
      else:
        data.lux += ru[i,:].T * rx[i,:] * self._inf
        return data.lux  # it doesn't make sense to continue
    return data.lux


class StateBarrier(XLogBarrier):
  """ Log-barrier for state bounds.

  Here the state bounds as the form: lb < x < ub where lb and ub are the lower
  and upper bounds. It's required that the lower and upper bounds have the
  same dimension.
  """

  def __init__(self, ub, lb):
    """ Construct the state barrier function.

    :param ub: upper bound
    :param lb: lower bound
    """
    assert ub.shape == lb.shape, \
      'The lower and upper bounds have different dimension.'
    assertGreaterThan(ub, lb)
    bound = np.vstack([ub, -lb])
    XLogBarrier.__init__(self, bound)

  def r(self, data, x, u):
    data.r[:data.n] = x
    data.r[data.n:] = -x
    return data.r

  def rx(self, data, x, u):
    data.rx[:data.n,:] = np.ones((data.n, data.n))
    data.rx[data.n:,:] = -np.ones((data.n, data.n))
    return data.rx


class ControlBarrier(ULogBarrier):
  """ Log-barrier for control bounds.

  Here the control bounds as the form: lb < u < ub where lb and ub are the lower
  and upper bounds. It's required that the lower and upper bounds have the
  same dimension.
  """

  def __init__(self, ub, lb):
    """ Construct the control barrier function.

    :param ub: upper bound
    :param lb: lower bound
    """
    assert ub.shape == lb.shape, \
      'The lower and upper bounds have different dimension.'
    assertGreaterThan(ub, lb)
    bound = np.vstack([ub, -lb])
    ULogBarrier.__init__(self, bound)

  def r(self, data, x, u):
    data.r[:data.m] = u
    data.r[data.m:] = -u
    return data.r

  def ru(self, data, x, u):
    data.ru[:data.m,:] = np.ones((data.m, data.m))
    data.ru[data.m:,:] = -np.ones((data.m, data.m))
    return data.ru