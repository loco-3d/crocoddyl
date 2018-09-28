import abc
import math
import numpy as np
from cddp.utils import assertClass


class DynamicalSystem(object):
  """ This abstract class declares virtual methods for defining the system
  evolution and its derivatives.

  It allows us to define any kind of smooth system dynamics of the form
  v = f(x,u). The vector field v belongs to the tangent space TxQ of given
  configuration point x on the configuration manifold Q. The control (input)
  vector u allows the system to evolves toward a desired configuration point.
  The dimension of the configuration space Q, its tanget space TxQ and the
  control (input) vector are nq, nv and m, respectively.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, nq, nv, m, integrator, discretizer):
    """ Construct the dynamics model.

    :param nq: dimension of the configuration space Q
    :param nv: dimension of the tangent bundle over the configuration space TQ
    :param m: dimension of the control input
    """
    assertClass(integrator, 'Integrator')
    assertClass(discretizer, 'Discretizer')

    self.nq = nq
    self.nv = nv
    self.m = m
    self.integrator = integrator
    self.discretizer = discretizer

    # Creates internally the integrator and discretizer data
    self.integrator.createData(nq, nv)
    self.discretizer.createData(nv)

  def createData(self):
    """ Create the system dynamics data.
    """
    from cddp.data.system import SystemData
    return SystemData(self.nq, self.nv, self.m)

  def stepForward(self, data, x, u, dt):
    """ Compute the next state value

    :param data: system data
    :param x: configuration point
    :param u: control vector
    :param dt: integration step
    """
    # Integrate the time-continuos dynamics in order to get the next state
    # value
    return self.integrator(self, data, x, u, dt)
  
  def computeDerivatives(self, data, x, u, dt):
    """ Compute the discrete-time derivatives of dynamics

    :param data: system data
    :param x: configuration point
    :param u: control vector
    :param dt: integration step
    """
    # Computing the time-continuos linearized system, i.e. dv = fx*dx + fu*du,
    # and converting it into discrete one
    self.discretizer(self, data, x, u, dt)
    return data.fx, data.fu

  @abc.abstractmethod
  def f(self, data, x, u):
    """ Evaluate the evolution function and stores the result in data.

    :param data: system data
    :param x: configuration point
    :param u: control input
    :returns: generalized velocity in x configuration
    """
    pass

  @abc.abstractmethod
  def fx(self, data, x, u):
    """ Evaluate the system Jacobian w.r.t. the configuration point and stores
    the result in data.

    :param data: system data
    :param x: configuration point
    :param u: control input
    :returns: system Jacobian w.r.t the configuration point
    """
    pass

  @abc.abstractmethod
  def fu(self, data, x, u):
    """ Evaluate the system Jacobian w.r.t. the control and stores the result
    in data.

    :param data: system data
    :param x: configuration point
    :param u: control input
    :returns: system Jacobian w.r.t the control
    """
    pass

  @abc.abstractmethod
  def advanceConfiguration(self, x, dx):
    """ Operator that advances the configuration state

    :param x: configuration point
    :param dx: displacement in tangent space of configuration manifold
    :returns: next configuration point
    """
    pass

  @abc.abstractmethod
  def differenceConfiguration(self, x_next, x_curr):
    """ Operator that differentiates the configuration state.

    :param x_next: next configuration point
    :param x_curr: current configuration point
    """
    # return xf - x0
    pass

  def getConfigurationDimension(self):
    """ Get the configuration space dimension.

    :returns: dimension of configuration space
    """
    return self.nq

  def getTangentDimension(self):
    """ Get the tangent bundle dimension.

    :returns: dimension of tangent bundle of the configuration space
    """
    return self.nv

  def getControlDimension(self):
    """ Get the control dimension.

    :returns: dimension of the control vector
    """
    return self.m



class NumDiffDynamicalSystem(DynamicalSystem):
  """ This abstract class declares virtual methods for defining the system
  evolution where its derivatives are computed numerically.

  This class uses numerical differentiation for computing the state and control
  derivatives of a dynamic model.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, nq, nv, m, integrator, discretizer):
    """ Construct the dynamics model.

    :param nq: dimension of the configuration manifold
    :param nv: dimension of the tangent space of the configuration manifold
    :param m: dimension of the control space
    """
    DynamicalSystem.__init__(self, nq, nv, m, integrator, discretizer)
    self.sqrt_eps = math.sqrt(np.finfo(float).eps)
    self.f_nom = np.matrix(np.zeros((nv, 1)))

  @abc.abstractmethod
  def advanceConfiguration(self, x, dx):
    """ Operator that advances the configuration state

    :param x: configuration point
    :param dx: displacement in tangent space of configuration manifold
    :returns: next configuration point
    """
    pass

  @abc.abstractmethod
  def differenceConfiguration(self, x_next, x_curr):
    """ Operator that differentiates the configuration state.

    :param x_next: next configuration point
    :param x_curr: current configuration point
    """
    pass

  def fx(self, data, x, u):
    """ Compute numerically the system Jacobian w.r.t. the configuration point
    and stores the result in data.

    :param data: system data
    :param x: configuration state
    :param u: control input
    :returns: system Jacobian w.r.t. the configuration point
    """
    np.copyto(self.f_nom, self.f(data, x, u))
    for i in range(data.nv):
      v_pert = np.zeros((data.nv, 1))
      v_pert[i] += self.sqrt_eps
      x_pert = self.advanceConfiguration(x.copy(), v_pert)
      data.fx[:, i] = (self.f(data, x_pert, u).copy() - self.f_nom) / self.sqrt_eps
    np.copyto(data.f, self.f_nom)
    return data.fx

  def fu(self, data, x, u):
    """ Compute numerically the system Jacobian w.r.t. the control and stores
    the result in data.

    :param data: system data
    :param x: configuration state
    :param u: control input
    :returns: system Jacobian w.r.t. the control
    """
    np.copyto(self.f_nom, self.f(data, x, u))
    for i in range(data.m):
      u_pert = u.copy()
      u_pert[i] += self.sqrt_eps
      data.fu[:, i] = (self.f(data, x, u_pert).copy() - self.f_nom) / self.sqrt_eps
    np.copyto(data.f, self.f_nom)
    return data.fu