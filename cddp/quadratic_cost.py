from cost import TerminalCost, RunningCost, TerminalResidualCost, RunningResidualCost
import numpy as np
import abc


class TerminalQuadraticCost(TerminalCost):
  """ This abstract class creates a quadratic terminal cost of the form:
  0.5 xr^T Q xr.

  An important remark here is that the state residual (i.e. xr) depends linearly
  on the state. This cost function can be used to define a goal state penalty.
  This residual has to be implemented in a derived class (i.e. xr function).

  Before computing the cost values, it is needed to set up the Q matrix. We
  define it as diagonal matrix because: 1) it is easy to tune and 2) we can
  exploit an efficient computation of it. Additionally, for efficiency
  computation, we assume that compute first l (or lx) whenever you want
  to compute lx (or lxx).
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    TerminalCost.__init__(self)

  def setWeights(self, q):
    """ Sets the diagonal value of the Q matrix.

    :param q: array of n elements, where n is the number of state variables
    """
    assert q.shape > 1, "The Q weights have to be described as an array. \
      We define it as diagonal matrix."
    assert (q > 0).all(), "The Q weights have to be positive."

    # Setting up the weight values
    self._q = q
    self._Q = np.diag(self._q.reshape(-1))

  def l(self, data, x):
    # The quadratic term is as follows 0.5 * xr^T Q xr. We compute it
    # efficiently by exploiting the fact thar Q is a diagonal matrix
    xr = self.xr(data, x)
    assert xr.shape == (data.n, 1), "The residual should have dimension \
      equals (" + str(data.n) + ",1)."
    data.l[0] = 0.5 * np.asscalar(xr.T * np.multiply(self._q[:, None], xr))
    return data.l[0]

  def lx(self, data, x):
    # The 1st derivative of a quadratic term is Q xr. We compute it efficiently
    # by exploiting the fact that Q is a diagonal matrix
    xr = data.xr  # for efficiency we assume that you run l() first
    np.copyto(data.lx, np.multiply(self._q[:, None], xr))
    return data.lx

  def lxx(self, data, x):
    # This derivatives has a constant value (i.e Q)
    np.copyto(data.lxx, self._Q)
    return data.lxx


class TerminalResidualQuadraticCost(TerminalResidualCost):
  """ This abstract class creates a quadratic terminal cost of the form:
  0.5 r^T Q r.

  An important remark here is that the state residual (i.e. r(x) e R^k) is a
  general function of the state with k dimension. This method aims to implement
  non-linear residual functions. In case of linear residuals, we suggest to use
  the TerminalQuadraticCost class for efficient computation. This cost function
  can be used to defined desired terminal features (e.g. center of mass, SE3
  trajectory, etc). The derived class has to implement the residual function
  and its derivatives (i.e. r, rx, ru).

  Before computing the cost values, it is needed to set up the Q matrix. We
  define it as diagonal matrix because: 1) it is easy to tune and 2) we can
  exploit an efficient computation of it. Additionally, for efficiency
  computation, we assume that compute first l (or lx) whenever you want to
  compute lx (or lxx).
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, k):
    TerminalResidualCost.__init__(self, k)

  def setWeights(self, q):
    """ Defines the Q matrix.

    :param q: array of n elements, where n is the number of state variables
    """
    assert q.shape > 1, "The Q weights have to be described as an array. \
      We define it as diagonal matrix."
    assert len(q) == self.k, "Wrong dimension of the Q array, it should be "\
      + str(self.k) + "."
    assert (q > 0).all(), "The Q weights have to be positive."

    # Setting up the weight values
    self._q = q

  def l(self, data, x):
    # The quadratic term is as follows 0.5 * r^T Q r. We compute it efficiently
    # by exploiting the fact that Q is a diagonal matrix
    r = self.r(data, x)
    assert r.shape == (data.k, 1), "The residual should have dimension equals \
      (" + str(data.k) + ",1)."
    data.l[0] = 0.5 * np.asscalar(r.T * np.multiply(self._q[:, None], r))
    return data.l[0]

  def lx(self, data, x):
    # The 1st derivative of a quadratic term is rx^T Q r. We compute it
    # efficiently by exploiting the fact that Q is a diagonal matrix
    r = data.r  # for efficiency we assume that you run l() first
    rx = self.rx(data, x)
    assert rx.shape == (data.k, data.n), "The residual Jacobian should have \
      dimension equals (" + str(data.k) + "," + str(data.n) + ")."
    np.copyto(data.lx, rx.T * np.multiply(self._q[:, None], r))
    return data.lx

  def lxx(self, data, x):
    # The 2nd derivative of a quadratic term is rxx^T Q r + rx^T Q rx. Note that
    # we approximate it using Gauss (i.e. neglecting the residual Hessian).
    # Additionally, we compute it efficiently by exploiting the fact that Q is
    # a diagonal matrix
    rx = data.rx  # for efficiency we assume that you run lxx() first
    np.copyto(data.lxx, rx.T * np.multiply(self._q[:, None], rx))
    return data.lxx


class RunningQuadraticCost(RunningCost):
  """ This abstract class creates a quadratic running cost of the form:
  0.5 (xr^T Q xr + ur^t R ur).

  An important remark here is that the state and control residuals (i.e. xr and
  ur) depend linearly on the state and control, respectively. This cost function
  can be used for describing a tracking error, or regularization, in the state
  and/or control vectors. The residuals has to be implemented in a derived class
  (i.e. xr, ur functions).

  Before computing the cost values, it is needed to set up the Q and R
  matrices. We define those matrices as diagonal ones because: 1) it is easy to
  tune and 2) we can exploit an efficient computation of them. Additionally,
  for efficiency computation, we assume that compute first l (or lx, lu)
  whenever you want to compute lx and lu (or lxx, luu, lux).
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    RunningCost.__init__(self)

  def setWeights(self, q, r):
    """ Defines the Q and R weighting matrices.

    :param q: array of n elements, where n is the number of state variables
    :param r: array of m elements, where n is the number of control variables
    """
    assert q.shape > 1, "The Q weights have to be described as an array. We \
      define it as diagonal matrix."
    assert r.shape > 1, "The R weights have to be described as an array. We \
      define it as diagonal matrix."
    assert (q > 0).all(), "The Q weights have to be positive."
    assert (r > 0).all(), "The R weights have to be positive."

    # Setting up the weight values
    self._q = q
    self._r = r
    self._Q = np.diag(self._q.reshape(-1))
    self._R = np.diag(self._r.reshape(-1))

  def l(self, data, x, u):
    # The quadratic term is as follows 0.5 * (xr^T Q xr + ur^T R ur). We compute
    # it efficiently by exploiting the fact thar Q is a diagonal matrix
    xr = self.xr(data, x, u)
    ur = self.ur(data, x, u)
    assert xr.shape == (data.n, 1), "The state residual should have dimension \
      equals (" + str(data.n) + ",1)."
    assert ur.shape == (data.m, 1), "The residual should have dimension equals \
      (" + str(data.m) + ",1)."
    data.l[0] = 0.5 * (np.asscalar(xr.T * np.multiply(self._q[:, None], xr)) +
                       np.asscalar(ur.T * np.multiply(self._r[:, None], ur)))
    return data.l[0]

  def lx(self, data, x, u):
    # The 1st derivative of a quadratic term is Q xr. We compute it efficiently
    # by exploiting the fact thar Q is a diagonal matrix
    xr = data.xr  # for efficiency we assume that you run l() first
    np.copyto(data.lx, np.multiply(self._q[:, None], xr))
    return data.lx

  def lu(self, data, x, u):
    # The 1st derivative of a quadratic term is ru^T Q r. We compute it
    # efficiently by exploiting the fact thar Q is a diagonal matrix
    ur = data.ur  # for efficiency we assume that you run l() first
    np.copyto(data.lu, np.multiply(self._r[:, None], ur))
    return data.lu

  def lxx(self, data, x, u):
    # The 2nd derivative of a quadratic term is Q
    np.copyto(data.lxx, self._Q)
    return data.lxx

  def luu(self, data, x, u):
    # The 2nd derivative of a quadratic term is R
    np.copyto(data.luu, self._R)
    return data.luu

  def lux(self, data, x, u):
    # This derivative is zero, so we do anything.
    return data.luu


class RunningResidualQuadraticCost(RunningResidualCost):
  """ This abstract class creates a quadratic running cost of the form:
  0.5 r^T Q r.

  An important remark here is that the residual vector (i.e. r(x,u)) is a
  general function of the state and control. This method aims to implement
  non-linear residual functions. In case of linear residuals, we suggest to use
  the RunningQuadraticCost class for efficient computation. This cost function
  can be used to track features (e.g. center of mass, SE3 trajectory, etc). The
  derived class has to implement the residual function and its derivatives
  (i.e. r, rx, ru).

  Before computing the cost values, it is needed to set up the Q. We define it
  as diagonal matrix because: 1) it is easy to tune and 2) we can exploit an
  efficient computation of it. Additionally, for efficiency computation, we
  assume that compute first l (or lx, lu) whenever you want to compute lx and
  lu (or lxx, luu, lux).
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, k):
    RunningResidualCost.__init__(self, k)

  def setWeights(self, q):
    """ Defines the Q matrix.

    :param q: array of n elements, where n is the number of state variables
    """
    assert q.shape > 1, "The Q weights have to be described as an array. We \
      define it as diagonal matrix."
    assert len(q) == self.k, "Wrong dimension of the Q array, it should be " \
      + str(self.k) + "."
    assert (q > 0).all(), "The Q weights have to be positive."

    # Setting up the weight values
    self._q = q

  def l(self, data, x, u):
    # The quadratic term is as follows 0.5 * r^T Q r. We compute it efficiently
    # by exploiting the fact thar Q is a diagonal matrix
    r = self.r(data, x, u)
    assert r.shape == (data.k, 1), "The residual should have dimension equals \
      (" + str(data.k) + ",1)."
    data.l[0] = 0.5 * np.asscalar(r.T * np.multiply(self._q[:, None], r))
    return data.l[0]

  def lx(self, data, x, u):
    # The 1st derivative of a quadratic term is rx^T Q r. We compute it
    # efficiently by exploiting the fact thar Q is a diagonal matrix
    r = data.r  # for efficiency we assume that you run l() first
    rx = self.rx(data, x, u)
    assert rx.shape == (data.k, data.n), "The Jacobian residual w.r.t. the \
      state should have dimension equals (" + str(data.k) + "," + str(data.n) + ")."
    np.copyto(data.lx, rx.T * np.multiply(self._q[:, None], r))
    return data.lx

  def lu(self, data, x, u):
    # The 1st derivative of a quadratic term is ru^T Q r. We compute it
    # efficiently by exploiting the fact thar Q is a diagonal matrix
    r = data.r  # for efficiency we assume that you run l() first
    ru = self.ru(data, x, u)
    assert ru.shape == (data.k, data.m), "The Jacobian residual w.r.t. the \
      control should have dimension equals (" + str(data.k) + "," + str(data.m) + ")."
    np.copyto(data.lu, ru.T * np.multiply(self._q[:, None], r))
    return data.lu

  def lxx(self, data, x, u):
    # The 2nd derivative of a quadratic term is rxx^T Q r + rx^T Q rx. We
    # approximate it using Gauss (i.e. neglecting the residual Hessian).
    # Additionally, we compute it efficiently by exploiting the fact thar Q is
    # a diagonal matrix
    rx = data.rx  # for efficiency we assume that you run lx() first
    np.copyto(data.lxx, rx.T * np.multiply(self._q[:, None], rx))
    return data.lxx

  def luu(self, data, x, u):
    # The 2nd derivative of a quadratic term is ruu^T Q r + ru^T Q ru. We
    # approximate it using Gauss (i.e. neglecting the residual Hessian).
    # Additionally, we compute it efficiently by exploiting the fact thar Q is
    # a diagonal matrix
    ru = data.ru  # for efficiency we assume that you run lu() first
    np.copyto(data.luu, ru.T * np.multiply(self._q[:, None], ru))
    return data.luu

  def lux(self, data, x, u):
    # The 2nd derivative of a quadratic term is rux^T Q r + ru^T Q rx. We
    # approximate it using Gauss (i.e. neglecting the residual Hessian).
    # Additionally, we compute it efficiently by exploiting the fact thar Q is
    # a diagonal matrix
    rx = data.rx  # for efficiency we assume that you run lx() first
    ru = data.ru  # for efficiency we assume that you run lu() first
    np.copyto(data.lux, ru.T * np.multiply(self._q[:, None], rx))
    return data.lux
