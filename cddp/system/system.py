import abc
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
    self.integrator.createData(nv)
    self.discretizer.createData(nv)

  def createData(self):
    """ Create the system dynamics data.
    """
    from cddp.data.system import DynamicalSystemData
    return DynamicalSystemData(self.nq, self.nv, self.m)

  def stepForward(self, data, x, u, dt):
    """ Compute the next state value

    :param data: dynamic model data
    :param x: configuration point
    :param u: control vector
    :param dt: integration step
    """
    # Integrate the time-continuos dynamics in order to get the next state
    # value
    return self.integrator(self, data, x, u, dt)
  
  def computeDerivatives(self, data, x, u, dt):
    """ Compute the discrete-time derivatives of dynamics

    :param data: dynamic model data
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

    :param data: dynamics data
    :param x: configuration point
    :param u: control input
    :returns: generalized velocity in x configuration
    """
    pass

  @abc.abstractmethod
  def fx(self, data, x, u):
    """ Evaluate the system Jacobian w.r.t. the configuration point and stores
    the result in data.

    :param data: dynamics data
    :param x: configuration point
    :param u: control input
    :returns: system Jacobian w.r.t the configuration point
    """
    pass

  @abc.abstractmethod
  def fu(self, data, x, u):
    """ Evaluate the system Jacobian w.r.t. the control and stores the result
    in data.

    :param data: dynamics data
    :param x: configuration point
    :param u: control input
    :returns: system Jacobian w.r.t the control
    """
    pass

  def stateDifference(self, xf, x0):
    """ Get the state different between xf and x0 (i.e. xf - x0).

    :param xf: configuration point
    :param x0: configuration point
    """
    return xf - x0

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