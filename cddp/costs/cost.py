import abc
import numpy as np

class XCost(object):
  """ This abstract class declares virtual methods for computing the terminal
  cost value and its derivatives.

  The running cost depends on the state vectors, which it has n values.
  """
  __metaclass__ = abc.ABCMeta
  
  def __init__(self, dynamicsModel, ref, weight):
    self.dynamicsModel = dynamicsModel
    self.ref = ref
    self.weight = weight

    # Creating the data structure of the cost and its derivatives w.r.t. the 
    # state
    self._l = 0.
    self._lx = np.zeros((2*dynamicsModel.nv(), 1))
    self._lxx = np.zeros((2*self.dynamicsModel.nv(), 2*self.dynamicsModel.nv()))

  @abc.abstractmethod
  def forwardTerminalCalc(self, dynamicsData):
    pass

  @abc.abstractmethod
  def backwardTerminalCalc(self, dynamicsData):
    pass

  def getl(self):
    return self._l

  def getlx(self):
    return self._lx

  def getlxx(self):
    return self._lxx


class XUCost(XCost):
  def __init__(self, dynamicsModel, ref, weight):
    XCost.__init__(self, dynamicsModel, ref, weight)

    # Creating the data structure of the cost derivatives w.r.t. the control
    self._lu = np.zeros((dynamicsModel.nu(), 1))
    self._luu = np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nu()))
    self._lux = np.zeros((self.dynamicsModel.nu(), 2*self.dynamicsModel.nv()))

  @abc.abstractmethod
  def forwardRunningCalc(self, dynamicsData):
    pass

  @abc.abstractmethod
  def backwardRunningCalc(self, dynamicsData):
    pass

  def getlu(self):
    return self._lu

  def getluu(self):
    return self._luu

  def getlux(self):
    return self._lux


class QuadraticCost(XUCost):
  """This abstract class creates a quadratic cost of the form:
  0.5 xr^T Q xr.

  An important remark here is that the state residual (i.e. xr) depends linearly
  on the state. This cost function can be used to define a goal state penalty.
  This residual has to be implemented in a derived class (i.e. xr function).

  Before computing the cost values, it is needed to set up the Q matrix. We
  define it as diagonal matrix because: 1) it is easy to tune and 2) we can
  exploit an efficient computation of it. Additionally, for efficiency
  computation, we assume that compute first l (or lx) whenever you want
  to compute lx (or lxx)."""

  __metaclass__=abc.ABCMeta

  def __init__(self, dynamicsModel, ref, weight, dim):
    XUCost.__init__(self, dynamicsModel, ref, weight)

    # Residual dimension
    self.nr = dim

    # Creating the data structure of the residual and its derivatives
    self._r = np.zeros((self.nr, 1))
    self._rx = np.zeros((self.nr, 2*self.dynamicsModel.nv()))
    self._ru = np.zeros((self.nr, self.dynamicsModel.nu()))

  @abc.abstractmethod
  def updateResidual(self, dynamicsData):
    pass

  @abc.abstractmethod
  def updateLineaResidualModel(self, dynamicsData):
    pass

  def forwardRunningCalc(self, dynamicsData):
    self.updateResidual(dynamicsData)

    # Computing the cost value
    self._l = 0.5 * np.asscalar(np.dot(self._r.T, np.multiply(self.weight, self._r)))

  def forwardTerminalCalc(self, dynamicsData):
    self.forwardRunningCalc(dynamicsData)

  def backwardRunningCalc(self, dynamicsData):
    self.updateLineaResidualModel(dynamicsData)

    # Computing the quadratic model of the cost function
    W_r = np.multiply(self.weight, self._r)
    W_rx = np.multiply(self.weight, self._rx)
    np.copyto(self._lx, np.dot(self._rx.T, W_r))
    np.copyto(self._lu, np.dot(self._ru.T, W_r))
    np.copyto(self._lxx, np.dot(self._rx.T, W_rx))
    np.copyto(self._luu, np.dot(self._ru.T, np.multiply(self.weight, self._ru)))
    np.copyto(self._lux, np.dot(self._ru.T, W_rx))

  def backwardTerminalCalc(self, dynamicsData):
    self.updateLineaResidualModel(dynamicsData)

    # Computing the quadratic model of the cost function
    W_r = np.multiply(self.weight, self._r)
    np.copyto(self._lx, np.dot(self._rx.T, W_r))
    np.copyto(self._lu, np.dot(self._ru.T, W_r))
    np.copyto(self._lxx, np.dot(self._rx.T, np.multiply(self.weight, self._rx)))