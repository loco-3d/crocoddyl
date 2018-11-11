import abc
import numpy as np


class TerminalCostData(object):
  """ Data structure for the terminal cost of a specific time interval.

  The terminal cost terms are the cost value and its derivates with respect to
  the state. The dimensions of these vector or matrices depends on the dynamic
  model.
  """
  def __init__(self, nx):
    """ Creates the terminal cost data.

    :param nx: state dimension
    """
    self.l = 0.
    self.lx = np.zeros((nx,1))
    self.lxx = np.zeros((nx,nx))


class RunningCostData(TerminalCostData):
  """ Data structure for the running cost of a specific time interval.

  The running cost terms includes the terminal ones plus its derivates
  with respect to the control. The dimensions of these vector or matrices
  depends on the dynamic
  model.
  """
  def __init__(self, nx, nu):
    """ Creates the running cost data.

    :param nx: state dimension
    :param nu: control dimension
    """
    TerminalCostData.__init__(self, nx)
    self.lu = np.zeros((nu,1))
    self.lux = np.zeros((nu,nx))
    self.luu = np.zeros((nu,nu))


class RunningCost(object):
  """ This abstract class declares virtual methods for computing the running
  cost value and its quadratic approximation.

  We consider a general cost function of the form: l(x), where x is the system's
  state which is defined by a vector that represent a point (R^{nx}) in the
  state space (a geometrical manifold). Instead the quadratic approximation has
  the form: lx*dx + dx^T*lxx*dx, where dx lies in the tangent space (R^{nv})
  around a nominal point in the geometrical manifold. Note that lx and lxx
  belong to R^{nv} and R^{nv\times nv}, respectively. Furthermore, a point x 
  can be describe more than nx tuples, e.g. SE(3) is a 6-dimensional manifold
  described with 12 tuples.
  """
  def __init__(self, nx, nu):
    """ Creates the running cost data.

    :param nx: state dimension
    :param nu: control dimension
    """
    # Creating the data structure of the cost and its quadratic approximantion.
    # Note that this approximation as the form:
    # [dx^T du^T]*[lx; lu] + [dx^T du^T]*[lxx lxu; lux luu]*[dx; du] where dx
    # lies in the tangent space (R^{nx}) around a nominal point in the
    # geometrical manifold, and du is point near to the nominal control (R^{nu})
    self._data = RunningCostData(nx, nu)

  @abc.abstractmethod
  def updateCost(self, dynamicsData):
    """ Update the cost value according an user-define function.

    The new cost value overwrites the internal data l.
    """
    pass

  @abc.abstractmethod
  def updateQuadraticAppr(self, dynamicsData):
    """ Update the quadratic approximation of the user-define cost function.

    The new quadratic approximation of the cost function overwrites the
    following internal data lx and lxx.
    """
    pass

  def getl(self):
    """ Return the current cost value.
    """
    return self._data.l

  def getlx(self):
    """ Return the current gradient of cost w.r.t. the state
    """
    return self._data.lx

  def getlu(self):
    """ Return the current gradient of cost w.r.t. the control
    """
    return self._data.lu

  def getlxx(self):
    """ Return the current Hessian of cost w.r.t. the state
    """
    return self._data.lxx

  def getluu(self):
    """ Return the current Hessian of cost w.r.t. the control
    """
    return self._data.luu

  def getlux(self):
    """ Return the current Hessian of cost w.r.t. the control and state
    """
    return self._data.lux


class TerminalCost(object):
  """ This abstract class declares virtual methods for computing the terminal
  cost value and its quadratic approximation.

  We consider a general cost function of the form: l(x), where x is the system's
  state which is defined by a vector that represent a point (R^{nx}) in the
  state space (a geometrical manifold). Instead the quadratic approximation has
  the form: lx*dx + dx^T*lxx*dx, where dx lies in the tangent space (R^{nx})
  around a nominal point in the geometrical manifold. Note that lx and lxx
  belong to R^{nx} and R^{nx\times nx}, respectively. Furthermore, a point x 
  can be describe more than nx tuples, e.g. SE(3) is a 6-dimensional manifold
  described with 12 tuples.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, nx):
    """ Creates the terminal cost data.

    :param nx: state dimension
    """
    # Creating the data structure of the cost and its quadratic approximantion.
    # Note that this approximation as the form: dx^T*lx + dx^T*lxx*dx, where dx
    # lies in the tangent space (R^{nx}) around a nominal point in the
    # geometrical manifold
    self._data = TerminalCostData(nx)

  @abc.abstractmethod
  def updateCost(self, dynamicsData):
    """ Update the cost value according an user-define function.

    The new cost value overwrites the internal data l.
    """
    pass

  @abc.abstractmethod
  def updateQuadraticAppr(self, dynamicsData):
    """ Update the quadratic approximantion of the user-define cost function.

    The new quadratic approximation of the cost function overwrites the
    following internal data lx and lxx.
    """
    pass

  def getl(self):
    """ Return the current cost value.
    """
    return self._data.l

  def getlx(self):
    """ Return the current gradient of cost w.r.t. the state
    """
    return self._data.lx

  def getlxx(self):
    """ Return the current Hessian of cost w.r.t. the state
    """
    return self._data.lxx


class RunningQuadraticCost(RunningCost):
  """ This abstract class creates a running quadratic cost of the form:
  0.5 r(x,u)^T Q r(x,u).
  
  It declares virtual methods for updating the residual function r(x,u) and its
  linear approximation. Both are needed for computing the cost function or its
  quadratic approximation.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, nx, nu, nr, weight):
    RunningCost.__init__(self, nx, nu)

    # Reference state and weight of the quadratic cost
    self.weight = weight

    # Creating the data structure of the residual and its derivatives
    self._r = np.zeros((nr,1))
    self._rx = np.zeros((nr,nx))
    self._ru = np.zeros((nr,nu))
    self._Q_r = np.zeros((nr,1))
    self._Q_rx = np.zeros((nr,nx))
    self._Q_ru = np.zeros((nr,nu))

  @abc.abstractmethod
  def updateResidual(self, dynamicsData):
    """ Update the residual value according an user-define function.

    The new residual value overwrites the internal data r.
    """
    pass

  @abc.abstractmethod
  def updateResidualLinearAppr(self, dynamicsData):
    """ Update the linear approximantion of the user-define residual function.

    The new linear approximation of the residual function overwrites the
    internal data rx. We neglect the Hessian of the residual (i.e. rxx = 0).
    """
    pass

  def updateCost(self, dynamicsData):
    # Updating the residual function value
    self.updateResidual(dynamicsData)

    # Updating the cost value
    self._data.l = \
      0.5 * np.asscalar(np.dot(self._r.T, np.multiply(self.weight, self._r)))

  def updateQuadraticAppr(self, dynamicsData):
    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(dynamicsData)

    # Updating the quadratic approximation of the cost function
    np.copyto(self._Q_r, np.multiply(self.weight, self._r))
    np.copyto(self._Q_rx, np.multiply(self.weight, self._rx))
    np.copyto(self._Q_ru, np.multiply(self.weight, self._ru))
    np.copyto(self._data.lx, np.dot(self._rx.T, self._Q_r))
    np.copyto(self._data.lu, np.dot(self._ru.T, self._Q_r))
    np.copyto(self._data.lxx, np.dot(self._rx.T, self._Q_rx))
    np.copyto(self._data.luu, np.dot(self._ru.T, self._Q_ru))
    np.copyto(self._data.lux, np.dot(self._ru.T, self._Q_rx))


class TerminalQuadraticCost(TerminalCost):
  """ This abstract class creates a running quadratic cost of the form:
  0.5 r(x)^T Q r(x).
  
  It declares virtual methods for updating the residual function r(x) and its
  linear approximation. Both are needed for computing the cost function or its
  quadratic approximation.
  """
  def __init__(self, nx, nr, weight):
    TerminalCost.__init__(self, nx)

    # Weight of the quadratic cost
    self.weight = weight

    # Creating the data structure of the residual and its derivatives
    self._r = np.zeros((nr,1))
    self._rx = np.zeros((nr,nx))
    self._Q_r = np.zeros((nr,1))
    self._Q_rx = np.zeros((nr,nx))

  @abc.abstractmethod
  def updateResidual(self, dynamicsData):
    """ Update the residual value according an user-define function.

    The new residual value overwrites the internal data r.
    """
    pass

  @abc.abstractmethod
  def updateResidualLinearAppr(self, dynamicsData):
    """ Update the linear approximantion of the user-define residual function.

    The new linear approximation of the residual function overwrites the
    internal data rx. We neglect the Hessian of the residual (i.e. rxx = 0).
    """
    pass

  def updateCost(self, dynamicsData):
    # Updating the residual function value
    self.updateResidual(dynamicsData)

    # Updating the cost value
    self._data.l = \
      0.5 * np.asscalar(np.dot(self._r.T, np.multiply(self.weight, self._r)))

  def updateQuadraticAppr(self, dynamicsData):
    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(dynamicsData)

    # Updating the quadratic approximation of the cost function
    np.copyto(self._Q_r, np.multiply(self.weight, self._r))
    np.copyto(self._Q_rx, np.multiply(self.weight, self._rx))
    np.copyto(self._data.lx, np.dot(self._rx.T, self._Q_r)) 
    np.copyto(self._data.lxx, np.dot(self._rx.T, self._Q_rx))