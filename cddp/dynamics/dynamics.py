import abc
import numpy as np


class DynamicsData(object):
  """ This abstract class declares virtual methods for updating the dynamics
  and its linear approximation.

  We consider a general dynamic function of the form: d/dt([q; v]) = [v; a(q,v,u)]
  where q is the configuration point (R^{nq}), v is its tangent velocity (R^nv),
  u is the control vector (R^{nu}) and the system's state is x = [q; v]. The 
  a() describes the acceleration evolution of the system. Note that in general
  q could be described with a number of tuples higher than nv (i.e. nq >= nv).
  For instance, if q is a point in a SE(3) manifold then we need 12 tuples (or 7
  tuples for quaternion-based description) to describe it.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, dynamicsModel, dt):
    # Current and previous state and control
    self.x = np.zeros((dynamicsModel.nxImpl(), 1))
    self.u = np.zeros((dynamicsModel.nu(), 1))
    self.x_prev = np.zeros((dynamicsModel.nxImpl(), 1))
    self.u_prev = np.zeros((dynamicsModel.nu(), 1))

    # Terms for linear approximantion, which has the form:
    #   d/dt([q; v]) = [0, I; aq, av]*[q; v] + [0; au]*u
    self.aq = np.zeros((dynamicsModel.nv(), dynamicsModel.nv()))
    self.av = np.zeros((dynamicsModel.nv(), dynamicsModel.nv()))
    self.au = np.zeros((dynamicsModel.nv(), dynamicsModel.nu()))

    # Creating the discretizer data
    if dt != 0.:
      self.discretizer = dynamicsModel.discretizer.createData(dynamicsModel, dt)

    self.diff_x = np.zeros((dynamicsModel.nx(), 1))


class DynamicsModel(object):
  "Base class to define the dynamics model"
  __metaclass__ = abc.ABCMeta
  
  def __init__(self, nxImpl, nx, nu):
    self._nx_impl = nxImpl
    self._nx = nx
    self._nu = nu

  @staticmethod
  def forwardRunningCalc(dynamicsModel, dynamicsData):
    "implement compute all terms for forward pass"
    pass

  @staticmethod
  def forwardTerminalCalc(dynamicsModel, dynamicsData):
    "implement compute all terms for forward pass"
    pass

  @staticmethod
  def backwardRunningCalc(dynamicsModel, dynamicsData):
    "implement compute all terms for backward pass"
    pass

  @staticmethod
  def backwardTerminalCalc(dynamicsModel, dynamicsData):
    "implement compute all terms for backward pass"
    pass

  @abc.abstractmethod
  def createData(dynamicsModel, tInit, dt):
    pass

  @staticmethod
  def deltaX(dynamicsModel, dynamicsData, x0, x1):
    pass

  def nxImpl(self):
    return self._nx_impl

  def nx(self):
    return self._nx

  def nu(self):
    return self._nu