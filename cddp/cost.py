import abc


class XCost(object):
  """ This abstract class declares virtual methods for computing the terminal cost value and
  its derivatives.

  The running cost depends on the state vectors, which it has n values.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def createData(self, n):
    """ Creates the terminal cost data structure.

    :param n: dimension of the state
    """
    pass

  @abc.abstractmethod
  def l(self, data, x):
    """ Evaluates the terminal cost and stores the result in data.

    :param data: terminal cost data
    :param x: state vector
    :return: terminal cost
    """
    pass

  @abc.abstractmethod
  def lx(self, data, x):
    """ Evaluates the Jacobian of the terminal cost and stores the result in data.

    :param data: terminal cost data
    :param x: state vector
    :return: Jacobian of the terminal cost
    """
    pass

  @abc.abstractmethod
  def lxx(self, data, x):
    """ Evaluates the Hessian of the terminal cost and stores the result in data.

    :param data: terminal cost data
    :param x: state vector
    :return: Hessian of the terminal cost
    """
    pass


class XUCost(object):
  """ This abstract class declares virtual methods for computing the running cost value and
  its derivatives.

  The running cost depends on the state and control vectors, those vectors have n and m dimensions,
  respectively.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def createData(self, n, m):
    """ Creates the terminal cost data structure

    :param n: dimension of the state
    :param m: dimensions of the control
    """
    pass

  @abc.abstractmethod
  def l(self, data, x, u):
    """ Evaluates the running cost and stores the result in data.

    :param data: running cost data
    :param x: state vector
    :param u: control vector
    :return: running cost
    """
    pass

  @abc.abstractmethod
  def lx(self, data, x, u):
    """ Evaluates the Jacobian of the running cost w.r.t. the state and stores the result in data.

    :param data: running cost data
    :param x: state vector
    :param u: control vector
    :return: Jacobian of the running cost with respect to the state
    """
    pass

  @abc.abstractmethod
  def lu(self, data, x, u):
    """ Evaluates the Jacobian of the running cost w.r.t. the control and stores the result in data.

    :param data: running cost data
    :param x: state vector
    :param u: control vector
    :return: Jacobian of the residual vector w.r.t. the control
    """
    pass

  @abc.abstractmethod
  def lxx(self, data, x, u):
    """ Evaluates the Hessian of the running cost w.r.t. the state and stores the result in data.

    :param data: running cost data
    :param x: state vector
    :param u: control vector
    :return: Hessian of the running cost w.r.t. the state
    """
    pass

  @abc.abstractmethod
  def luu(self, data, x, u):
    """ Evaluates the Hessian of the running cost w.r.t. the control and stores the result in data.

    :param data: running cost data
    :param x: state vector
    :param u: control vector
    :return: Hessian of the running cost w.r.t. the control
    """
    pass

  @abc.abstractmethod
  def lux(self, data, x, u):
    """ Evaluates the running cost derivatives w.r.t. the control and state and stores the result in data.

    :param data: running cost data
    :param x: state vector
    :param u: control vector
    :return: derivatives of the running cost w.r.t. the state and control
    """
    pass


class TerminalCost(XCost):
  """ This abstract class declares virtual methods for computing the terminal cost value and
  its derivatives.

  An important remark here is that the terminal cost is computed from linear residual vector.
  """
  __metaclass__ = abc.ABCMeta

  def createData(self, n):
    """ Creates the terminal cost data structure.

    :param n: dimension of the state
    """
    from data import TerminalCostData
    return TerminalCostData(n)

  @abc.abstractmethod
  def xr(self, data, x):
    """ Evaluates the residual vector and stores the result in data.

    :param data: terminal cost data
    :param x: state vector
    :return: state-residual vector
    """
    pass


class TerminalResidualCost(XCost):
  """ This abstract class declares virtual methods for computing the terminal cost value and
  its derivatives.

  An important remark here is that the terminal cost is computed from general residual vector r.
  Therefore, compared with the TerminalCost class, it is needed additionally to provide
  information about the residual derivatives (i.e. rx and ru)
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, k):
    # Residual vector dimension
    self.k = k

  def createData(self, n):
    """ Creates the terminal cost data structure and its residual.

    :param n: dimension of the state
    """
    from data import TerminalResidualCostData
    return TerminalResidualCostData(n, self.k)

  @abc.abstractmethod
  def r(self, data, x):
    """ Evaluates the residual vector and stores the result in data.

    :param data: terminal cost data
    :param x: state vector
    :return: residual vector
    """
    pass

  @abc.abstractmethod
  def rx(self, data, x):
    """ Evaluates the Jacobian of the residual vector and stores the result in data.

    :param data: terminal cost data
    :param x: state vector
    :return: Jacobian of the residual vector
    """
    pass


class RunningCost(XUCost):
  """ This abstract class declares virtual methods for computing the running cost value and
  its derivatives.

  An important remark here is that the running cost is computed from linear residual vector
  on the state (xr) and control (ur).
  """
  __metaclass__ = abc.ABCMeta

  def createData(self, n, m):
    """
    :param n: dimension of the state
    :param m: dimensions of the control
    """
    from data import RunningCostData
    return RunningCostData(n, m)

  @abc.abstractmethod
  def xr(self, data, x, u):
    """ Evaluates the residual state vector and stores the result in data.

    :param data: running cost data
    :param x: state vector
    :param u: control vector
    :return: state-residual vector
    """
    pass

  @abc.abstractmethod
  def ur(self, data, x, u):
    """ Evaluates the residual control vector and stores the result in data.

    :param data: running cost data
    :param x: state vector
    :param u: control vector
    :return: control-residual vector
    """
    pass


class RunningResidualCost(XUCost):
  """ This abstract class declares virtual methods for computing the running cost value and
  its derivatives.

  An important remark here is that the running cost is computed from general residual vector
  on the state and control, i.e. r(x,u). Therefore, compared with the RunningCost class, it is
  additionally needed to provide the information of the residual derivatives (i.e. rx, ru).
  The residual Hessians (rxx, ruu and rux) are neglected which is common in Gauss-Newton steps.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, k):
    # Residual vector dimension
    self.k = k

  def createData(self, n, m):
    """ Creates the data structure for the running cost and its residual.

    :param n: dimension of the state
    :param m: dimensions of the control
    """
    from data import RunningResidualCostData
    return RunningResidualCostData(n, m, self.k)

  @abc.abstractmethod
  def r(self, data, x, u):
    """ Evaluates the residual vector and stores the result in data.

    :param data: running cost data
    :param x: state vector
    :param u: control vector
    :return: residual vector
    """
    pass

  @abc.abstractmethod
  def rx(self, data, x, u):
    """ Evaluates the Jacobian of the residual vector w.r.t. the state and stores the result in data.

    :param data: running cost data
    :param x: state vector
    :param u: control vector
    :return: Jacobian of the residual vector w.r.t. the state
    """
    pass

  @abc.abstractmethod
  def ru(self, data, x, u):
    """ Evaluates the Jacobian of the residual vector w.r.t. the control and stores the result in data.

    :param data: running cost data
    :param x: state vector
    :param u: control vector
    :return: Jacobian of the residual vector w.r.t. the control
    """
    pass
