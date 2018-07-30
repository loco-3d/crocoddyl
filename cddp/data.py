import numpy as np
import abc


class XCostData(object):
  """ Data structure for state-dependent cost functions.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, n):
    """ Construct the data structure for only state-based cost functions.

    It requires the dimension of the state space or tangent manifold in case of
    diffeomorphism systems.
    :param n: state or tangent manifold dimension
    """
    # State dimension
    self.n = n

    # Creating the data structure of state cost function
    # and its derivatives
    self.l = np.matrix(np.zeros(1))
    self.lx = np.matrix(np.zeros((self.n, 1)))
    self.lxx = np.matrix(np.zeros((self.n, self.n)))


class XUCostData(object):
  """ Data structure for state/control-dependent cost functions.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, n, m):
    """ Construct the data structure for cost functions.

    It requires the dimension of the state and control spaces. In case of
    diffeomorphism systems, it's needed the dimension of the tangent space
    instead of the state one.
    :param n: state or tangent manifold dimension
    :param m: control dimension
    """
    # State and control dimension
    self.n = n
    self.m = m

    # Creating the data structure of state-control cost function and its
    # derivatives
    self.l = np.matrix(np.zeros(1))
    self.lx = np.matrix(np.zeros((self.n, 1)))
    self.lu = np.matrix(np.zeros((self.m, 1)))
    self.lxx = np.matrix(np.zeros((self.n, self.n)))
    self.luu = np.matrix(np.zeros((self.m, self.m)))
    self.lux = np.matrix(np.zeros((self.m, self.n)))


class TerminalCostData(XCostData):
  """ Data structure for terminal costs with a linear residual function.

  The residual is n-dimensional because it depends linearly on the state.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, n):
    """ Construct the data structure for only state-based cost functions.

    It requires the dimension of the state space or tangent manifold in case of
    diffeomorphism systems.
    :param n: state or tangent manifold dimension
    """
    # Creating the state-related data
    XCostData.__init__(self, n)

    # Creating the state residual and its derivatives
    self.xr = np.matrix(np.zeros((self.n, 1)))


class TerminalResidualCostData(XCostData):
  """ Data structure for terminal costs with a general residual function.

  In general, the residual is k-dimensional, its dimension depends on the
  implementation of this residual.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, n, k):
    """ Construct the data structure for only state-based cost functions

    It requires the dimension of the state space, or tangent manifold in case
    of diffeomorphism systems, and the dimension of the residual vector.
    :param n: state or tangent manifold dimension
    """
    # Creating the state-related data
    XCostData.__init__(self, n)

    # Residual dimension. Note that the residual is a general function
    # of the state
    self.k = k

    # Creating the data structure of the residual and its derivatives
    self.r = np.matrix(np.zeros((self.k, 1)))
    self.rx = np.matrix(np.zeros((self.k, self.n)))


class RunningCostData(XUCostData):
  """ Data structure for running costs with a linear residual function.

  The residual depends on the state and control. So we split it in the state
  and control residual. The state and control residual have a dimension of n 
  and m, respectively.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, n, m):
    """ Construct the data structure for cost functions.

    It requires the dimension of the state and control spaces. In case of
    diffeomorphism systems, it's needed the dimension of the tangent space
    instead of state one.
    :param n: state or tangent manifold dimension
    :param m: control dimension
    """
    # Creating the state-control cost data
    XUCostData.__init__(self, n, m)

    # Adding the state and control residual vectors
    self.xr = np.matrix(np.zeros((self.n, 1)))
    self.ur = np.matrix(np.zeros((self.m, 1)))


class RunningResidualCostData(XUCostData):
  """ Data structure for running costs with a general residual function.

  In general, the residual is k-dimensional and depends on the state and
  control. Its dimension is defined in the implementation of it.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, n, m, k):
    """ Construct the data structure for cost functions.

    It requires the dimension of the state and control spaces, and residual
    vector. In case of diffeomorphism systems, it's needed the dimension of the
    tangent space instead of the state one.
    :param n: state or tangent manifold dimension
    :param m: control dimension
    """
    # Creating the state-control cost data
    XUCostData.__init__(self, n, m)

    # Residual dimension
    self.k = k

    # Adding the residual and its derivatives
    self.r = np.matrix(np.zeros((self.k, 1)))
    self.rx = np.matrix(np.zeros((self.k, self.n)))
    self.ru = np.matrix(np.zeros((self.k, self.m)))


class CostManagerData():
  """ Data structure for the cost manager.

  The cost manager contains the total cost and a stack-of-cost (SoC) functions.
  It creates a dedicated data structure for the total cost, the sum of cost
  data, an another one for the SoC. The total cost and SoC can contained the
  data of the terminal or running costs.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    self.total = None
    self.soc = []


class DynamicsData(object):
  """ Data structure for the system dynamics.

  We only define the Jacobians of the dynamics, and not the Hessians of it,
  because our optimal controller by default uses the Gauss-Newton approximation.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, nq, nv, m):
    # Configuration and its tangent and control dimensions
    self.nq = nq
    self.nv = nv
    self.m = m

    # Creating the data structure for the ODE and its
    # derivative
    self.f = np.matrix(np.zeros((self.nv, 1)))
    self.fx = np.matrix(np.zeros((self.nv, self.nv)))
    self.fu = np.matrix(np.zeros((self.nv, self.m)))


class TerminalDDPData(object):
  """ Data structure for the terminal interval of the DDP.

  We create data for the nominal and new state values.
  """
  def __init__(self, dyn_data, cost_data):
    self.dynamics = dyn_data
    self.cost = cost_data

    # Configuration and tangent manifold dimension
    self.nq = self.dynamics.nq
    self.nv = self.dynamics.nv

    # Nominal and new state on the interval
    self.x = np.matrix(np.zeros((self.nq, 1)))
    self.x_new = np.matrix(np.zeros((self.nq, 1)))

    # Time of the terminal interval
    self.t = np.matrix(np.zeros(1))


class RunningDDPData(object):
  """ Data structure for the running interval of the DDP.

  We create data for the nominal and new state and control values. Additionally,
  this data structure contains regularized terms too (e.g. Quu_r).
  """
  def __init__(self, dyn_data, cost_data):
    self.dynamics = dyn_data
    self.cost = cost_data

    # Configuration manifold, tangent manifold and control dimensions
    self.nq = self.dynamics.nq
    self.nv = self.dynamics.nv
    self.m = self.dynamics.m

    # Nominal and new state on the interval
    self.x = np.matrix(np.zeros((self.nq, 1)))
    self.x_new = np.matrix(np.zeros((self.nq, 1)))

    # Nominal and new control command on the interval
    self.u = np.matrix(np.zeros((self.m, 1)))
    self.u_new = np.matrix(np.zeros((self.m, 1)))

    # Starting and final time on the interval
    self.t0 = np.matrix(np.zeros(1))
    self.tf = np.matrix(np.zeros(1))

    # Feedback and feedforward terms
    self.K = np.matrix(np.zeros((self.m, self.nv)))
    self.j = np.matrix(np.zeros((self.m, 1)))

    # Value function and its derivatives
    self.Vx = np.matrix(np.zeros((self.nv, 1)))
    self.Vxx = np.matrix(np.zeros((self.nv, self.nv)))

    # Quadratic approximation of the value function
    self.Qx = np.matrix(np.zeros((self.nv, 1)))
    self.Qu = np.matrix(np.zeros((self.m, 1)))
    self.Qxx = np.matrix(np.zeros((self.nv, self.nv)))
    self.Qux = np.matrix(np.zeros((self.m, self.nv)))
    self.Quu = np.matrix(np.zeros((self.m, self.m)))
    self.Qux_r = np.matrix(np.zeros((self.m, self.nv)))
    self.Quu_r = np.matrix(np.zeros((self.m, self.m)))