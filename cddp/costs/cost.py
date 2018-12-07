import abc
import numpy as np


class TerminalCostData(object):
  """ Data structure for the terminal cost of a specific time interval.

  The terminal cost terms are the cost value and its derivates with respect to
  the state. The dimensions of these vector or matrices depends on the dynamic
  model.
  """
  def __init__(self, dynamicsModel):
    """ Create the terminal cost data.

    :param dynamicsModel: dynamics model
    """
    # Creating the data structure of the cost and its quadratic approximantion.
    # Note that this approximation as the form: dx^T*lx + dx^T*lxx*dx, where dx
    # lies in the tangent space (R^{nx}) around a nominal point in the
    # geometrical manifold
    self.l = 0.
    self.lx = np.zeros((dynamicsModel.nx(),1))
    self.lxx = np.zeros((dynamicsModel.nx(),dynamicsModel.nx()))


class RunningCostData(TerminalCostData):
  """ Data structure for the running cost of a specific time interval.

  The running cost terms includes the terminal ones plus its derivates
  with respect to the control. The dimensions of these vector or matrices
  depends on the dynamic
  model.
  """
  def __init__(self, dynamicsModel):
    """ Creates the running cost data.

    :param dynamicsModel: dynamics model
    """
    # Creating the data structure of the cost and its quadratic approximantion.
    # Note that this approximation as the form:
    # [dx^T du^T]*[lx; lu] + [dx^T du^T]*[lxx lxu; lux luu]*[dx; du] where dx
    # lies in the tangent space (R^{nx}) around a nominal point in the
    # geometrical manifold, and du is point near to the nominal control (R^{nu})
    TerminalCostData.__init__(self, dynamicsModel)
    self.lu = np.zeros((dynamicsModel.nu(),1))
    self.lux = np.zeros((dynamicsModel.nu(),dynamicsModel.nx()))
    self.luu = np.zeros((dynamicsModel.nu(),dynamicsModel.nu()))


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
  def __init__(self):
    pass

  @abc.abstractmethod
  def createData(self, dynamicsModel):
    """ Create the running cost data.

    :param dynamicsModel: dynamics model
    """
    raise NotImplementedError("Not implemented yet.")

  @abc.abstractmethod
  def updateCost(self, costData, dynamicsData):
    """ Update the cost value according an user-defined cost function.

    The new cost value overwrites the internal data l.
    """
    raise NotImplementedError("Not implemented yet.")

  @abc.abstractmethod
  def updateQuadraticAppr(self, costData, dynamicsData):
    """ Update the quadratic approximation of the user-defined cost function.

    The new quadratic approximation of the cost function overwrites the
    following internal data lx and lxx.
    """
    raise NotImplementedError("Not implemented yet.")

  @abc.abstractmethod
  def setReference(costData, ref):
    """ Set the reference of the user-defined cost function

    :param ref: reference of the cost function
    """
    raise NotImplementedError("Not implemented yet.")

  @staticmethod
  def getl(costData):
    """ Return the current cost value.
    """
    return costData.l

  @staticmethod
  def getlx(costData):
    """ Return the current gradient of cost w.r.t. the state
    """
    return costData.lx

  @staticmethod
  def getlu(costData):
    """ Return the current gradient of cost w.r.t. the control
    """
    return costData.lu

  @staticmethod
  def getlxx(costData):
    """ Return the current Hessian of cost w.r.t. the state
    """
    return costData.lxx

  @staticmethod
  def getluu(costData):
    """ Return the current Hessian of cost w.r.t. the control
    """
    return costData.luu

  @staticmethod
  def getlux(costData):
    """ Return the current Hessian of cost w.r.t. the control and state
    """
    return costData.lux


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

  def __init__(self):
    pass

  @abc.abstractmethod
  def createData(self, dynamicsModel):
    """ Create the terminal cost data.

    :param dynamicsModel: dynamics model
    """
    raise NotImplementedError("Not implemented yet.")

  @abc.abstractmethod
  def updateCost(self, costData, dynamicsData):
    """ Update the cost value according an user-defined cost function.

    The new cost value overwrites the internal data l.
    """
    raise NotImplementedError("Not implemented yet.")

  @abc.abstractmethod
  def updateQuadraticAppr(self, costData, dynamicsData):
    """ Update the quadratic approximation of the user-defined cost function.

    The new quadratic approximation of the cost function overwrites the
    following internal data lx and lxx.
    """
    raise NotImplementedError("Not implemented yet.")

  @abc.abstractmethod
  def setReference(costData, ref):
    """ Set the reference of the user-defined cost function

    :param ref: reference of the cost function
    """
    raise NotImplementedError("Not implemented yet.")

  @staticmethod
  def getl(costData):
    """ Return the current cost value.
    """
    return costData.l

  @staticmethod
  def getlx(costData):
    """ Return the current gradient of cost w.r.t. the state
    """
    return costData.lx

  @staticmethod
  def getlxx(costData):
    """ Return the current Hessian of cost w.r.t. the state
    """
    return costData.lxx


class RunningQuadraticCostData(RunningCostData):
  def __init__(self, dynamicsModel, nr):
    # Creating the standard data structure of running cost
    RunningCostData.__init__(self, dynamicsModel)

    # Creating the data structure of the residual and its derivatives
    self.r = np.zeros((nr,1))
    self.rx = np.zeros((nr,dynamicsModel.nx()))
    self.ru = np.zeros((nr,dynamicsModel.nu()))
    self.Q_r = np.zeros((nr,1))
    self.Q_rx = np.zeros((nr,dynamicsModel.nx()))
    self.Q_ru = np.zeros((nr,dynamicsModel.nu()))


class TerminalQuadraticCostData(TerminalCostData):
  def __init__(self, dynamicsModel, nr):
    # Creating the standard data structure of terminal cost
    TerminalCostData.__init__(self, dynamicsModel)

    # Creating the data structure of the residual and its derivatives
    self.r = np.zeros((nr,1))
    self.rx = np.zeros((nr,dynamicsModel.nx()))
    self.Q_r = np.zeros((nr,1))
    self.Q_rx = np.zeros((nr,dynamicsModel.nx()))


class RunningQuadraticCost(RunningCost):
  """ This abstract class creates a running quadratic cost of the form:
  0.5 r(x,u)^T Q r(x,u).
  
  It declares virtual methods for updating the residual function r(x,u) and its
  linear approximation. Both are needed for computing the cost function or its
  quadratic approximation.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, nr, weight):
    RunningCost.__init__(self)

    # Dimension of the residual vector
    self.nr = nr

    # Reference state and weight of the quadratic cost
    self.weight = weight

  @abc.abstractmethod
  def updateResidual(self, costData, dynamicsData):
    """ Update the residual value according an user-define function.

    The new residual value overwrites the internal data r.
    """
    raise NotImplementedError("Not implemented yet.")

  @abc.abstractmethod
  def updateResidualLinearAppr(self, costData, dynamicsData):
    """ Update the linear approximantion of the user-define residual function.

    The new linear approximation of the residual function overwrites the
    internal data rx. We neglect the Hessian of the residual (i.e. rxx = 0).
    """
    raise NotImplementedError("Not implemented yet.")

  def updateCost(self, costData, dynamicsData):
    # Updating the residual function value
    self.updateResidual(costData, dynamicsData)

    # Updating the cost value
    costData.l = \
      0.5 * np.asscalar(np.dot(costData.r.T, np.multiply(self.weight, costData.r)))

  def updateQuadraticAppr(self, costData, dynamicsData):
    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(costData, dynamicsData)

    # Updating the quadratic approximation of the cost function
    np.copyto(costData.Q_r, np.multiply(self.weight, costData.r))
    np.copyto(costData.Q_rx, np.multiply(self.weight, costData.rx))
    np.copyto(costData.Q_ru, np.multiply(self.weight, costData.ru))
    np.copyto(costData.lx, np.dot(costData.rx.T, costData.Q_r))
    np.copyto(costData.lu, np.dot(costData.ru.T, costData.Q_r))
    np.copyto(costData.lxx, np.dot(costData.rx.T, costData.Q_rx))
    np.copyto(costData.luu, np.dot(costData.ru.T, costData.Q_ru))
    np.copyto(costData.lux, np.dot(costData.ru.T, costData.Q_rx))


class TerminalQuadraticCost(TerminalCost):
  """ This abstract class creates a running quadratic cost of the form:
  0.5 r(x)^T Q r(x).
  
  It declares virtual methods for updating the residual function r(x) and its
  linear approximation. Both are needed for computing the cost function or its
  quadratic approximation.
  """
  def __init__(self, nr, weight):
    TerminalCost.__init__(self)

    # Dimension of the residual vector
    self.nr = nr

    # Weight of the quadratic cost
    self.weight = weight

  @abc.abstractmethod
  def updateResidual(self, costData, dynamicsData):
    """ Update the residual value according an user-define function.

    The new residual value overwrites the internal data r.
    """
    raise NotImplementedError("Not implemented yet.")

  @abc.abstractmethod
  def updateResidualLinearAppr(self, costData, dynamicsData):
    """ Update the linear approximantion of the user-define residual function.

    The new linear approximation of the residual function overwrites the
    internal data rx. We neglect the Hessian of the residual (i.e. rxx = 0).
    """
    raise NotImplementedError("Not implemented yet.")

  def updateCost(self, costData, dynamicsData):
    # Updating the residual function value
    self.updateResidual(costData, dynamicsData)

    # Updating the cost value
    costData.l = \
      0.5 * np.asscalar(np.dot(costData.r.T, np.multiply(self.weight, costData.r)))

  def updateQuadraticAppr(self, costData, dynamicsData):
    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(costData, dynamicsData)

    # Updating the quadratic approximation of the cost function
    np.copyto(costData.Q_r, np.multiply(self.weight, costData.r))
    np.copyto(costData.Q_rx, np.multiply(self.weight, costData.rx))
    np.copyto(costData.lx, np.dot(costData.rx.T, costData.Q_r))
    np.copyto(costData.lxx, np.dot(costData.rx.T, costData.Q_rx))