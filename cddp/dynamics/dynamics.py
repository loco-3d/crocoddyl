import abc
import numpy as np


class DynamicsData(object):
  """ Basic data structure for the system dynamics.

  We consider a general system dynamics as: d/dt([q; v]) = [v; a(q,v,u)]
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
  """ This abstract class declares virtual methods for updating the dynamics
  and its linear approximation.

  We consider a general system dynamic as: d/dt([q; v]) = [v; a(q,v,u)]
  where q is the configuration point (R^{nq}), v is its tangent velocity (R^nv),
  u is the control vector (R^{nu}) and the system's state is x = [q; v]. The
  a() describes the acceleration evolution of the system. Note that in general
  q could be described with a number of tuples higher than nv (i.e. nq >= nv).
  For instance, if q is a point in a SE(3) manifold then we need 12 tuples (or 7
  tuples for quaternion-based description) to describe it.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, nq, nv, nu):
    """ Create the dynamic model.

    :param nq: number of tuples that describe the configuration point
    :param nv: dimension of the configuration space
    :param nu: dimension of control vector
    """
    self._nq = nq
    self._nv = nv
    self._nu = nu
    # Computing the dimension of the state space
    self._nx_impl = nq + nv
    self._nx = 2 * nv

  @abc.abstractmethod
  def createData(dynamicsModel, tInit, dt):
    """ Create the dynamics data.

    :param dynamicsModel: dynamics model
    :param tInit: starting time
    :param dt: step integration
    """
    pass

  @staticmethod
  def updateTerms(dynamicsModel, dynamicsData):
    """ Update the terms needed for an user-defined dynamics.

    :param dynamicsModel: dynamics model
    :param dynamicsData: dynamics data
    """
    pass

  @staticmethod
  def updateDynamics(dynamicsModel, dynamicsData):
    """ Update the an user-defined dynamics.

    :param dynamicsModel: dynamics model
    :param dynamicsData: dynamics data
    """
    pass

  @staticmethod
  def updateLinearAppr(dynamicsModel, dynamicsData):
    """ Update the an user-defined dynamics.

    :param dynamicsModel: dynamics model
    :param dynamicsData: dynamics data
    """
    pass

  @staticmethod
  def deltaX(dynamicsModel, dynamicsData, x0, x1):
    pass

  def forwardRunningCalc(dynamicsModel, dynamicsData):
    # Updating the dynamics
    dynamicsModel.updateDynamics(dynamicsData)

  def forwardTerminalCalc(dynamicsModel, dynamicsData):
    # Updating the dynamic terms
    dynamicsModel.updateTerms(dynamicsData)

  def backwardRunningCalc(dynamicsModel, dynamicsData):
    # Updating the continuous-time linear approximation
    dynamicsModel.updateLinearAppr(dynamicsData)
    # Discretizing this linear approximation
    dynamicsModel.discretizer.backwardRunningCalc(dynamicsModel, dynamicsData)

  def backwardTerminalCalc(dynamicsModel, dynamicsData):
    # Updating the dynamic terms
    dynamicsModel.updateTerms(dynamicsData)

  def nq(self):
    """ Return the number of tuples used to describe the configuration point.
    """
    return self._nq

  def nv(self):
    """ Return the dimension of the configuration space.
    """
    return self._nv

  def nu(self):
    """ Return the dimension of the control vector.
    """
    return self._nu

  def nxImpl(self):
    """ Return the number of tuples used to describe the state.
    """
    return self._nx_impl

  def nx(self):
    """ Return the dimension of the state space.
    """
    return self._nx