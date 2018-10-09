import abc
import numpy as np


class Integrator(object):
  """ This abstract class declares the virtual method for any integrator of the
  system dynamics.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    pass

  @abc.abstractmethod
  def createData(self, nq, nv):
    """ Create the data for the numerical integration of the system dynamics.

    Due to the integrator data is needed internally, we create inside the
    implementation class. It aims is to create data for the integrator.
    :param nq: dimension of the configuration manifold
    :param nv: dimension of the tangent space of the configuration manifold
    """
    pass

  @abc.abstractmethod
  def __call__(self, system, data, x, u, dt):
    """ Integrate numerically the system dynamics.

    This abstract method allows us to define integration rules of our ODE. It
    uses the system and data classes which defines the evolution function and
    its data, respectively. For more information about the system and data see
    the Dynamics class and and its data structure.
    :param system: system model
    :param data: system data
    :para x: configuration point
    :param u: control vector
    :param dt: integration step
    :returns: next state vector
    """
    pass


class GeometricIntegrator(object):
  """ This abstract class declares the virtual method for any integrator of a
  geometrical system.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    pass

  @abc.abstractmethod
  def createData(self, nq, nv):
    """ Create the data for the numerical integration of the system dynamics.

    Due to the integrator data is needed internally, we create inside the
    implementation class. It aims is to create data for the integrator.
    :param nq: dimension of the configuration manifold
    :param nv: dimension of the tangent space of the configuration manifold
    """
    pass

  @abc.abstractmethod
  def __call__(self, system, data, q, v, tau, dt):
    """ Integrate numerically the system dynamics.

    This abstract method allows us to define integration rules of our ODE. It
    uses the system and data classes which defines the evolution function and
    its data, respectively. For more information about the system and data see
    the Dynamics class and and its data structure.
    :param system: system model
    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: control vector
    :param dt: integration step
    :returns: next state vector
    """
    pass


class EulerIntegrator(Integrator):
  def createData(self, nq, nv):
    """ Create the internal data of forward Euler integrator of the system
    dynamics.

    :param nq: dimension of the configuration manifold
    :param nv: dimension of the tangent space of the configuration manifold
    """
    # No extra data needs to be created

  def __call__(self, system, data, x, u, dt):
    """ Integrate the system dynamics using the forward Euler scheme.

    :param system: system model
    :param data: system data
    :para x: configuration point
    :param u: control vector
    :param dt: sampling period
    :returns: next state vector
    """
    np.copyto(data.f,
             system.advanceConfiguration(x, dt * system.f(data, x, u)))
    return data.f


class GeometricEulerIntegrator(GeometricIntegrator):
  def createData(self, nq, nv):
    """ Create the internal data of forward Euler integrator of the system
    dynamics.

    :param nq: dimension of the configuration manifold
    :param nv: dimension of the tangent space of the configuration manifold
    """
    # No extra data needs to be created

  def __call__(self, system, data, q, v, tau, dt):
    """ Integrate the system dynamics using the forward Euler scheme.

    :param system: system model
    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: control vector
    :param dt: sampling period
    :returns: next state vector
    """
    data.f[data.nq:] = v + system.g(data, q, v, tau) * dt
    data.f[:data.nq] = \
      system.advanceConfiguration(q, data.f[data.nq:] * dt)
    return data.f


class RK4Integrator(Integrator):
  def createData(self, nq, nv):
    """ Create the internal data of forward Euler integrator of the system
    dynamics.

    :param nq: dimension of the configuration manifold
    :param nv: dimension of the tangent space of the configuration manifold
    """
    self.k1 = np.matrix(np.zeros((nv, 1)))
    self.k2 = np.matrix(np.zeros((nv, 1)))
    self.k3 = np.matrix(np.zeros((nv, 1)))
    self.k4 = np.matrix(np.zeros((nv, 1)))
    self.sum_k = np.matrix(np.zeros((nv, 1)))

  def __call__(self, system, data, x, u, dt):
    """ Integrate the system dynamics using the fourth-order Runge-Kutta method.

    :param system: system model
    :param data: system data
    :para x: state vector
    :param u: control vector
    :param dt: sampling period
    :returns: next state vector
    """
    np.copyto(self.k1, dt * system.f(data, x, u))
    np.copyto(self.k2, dt * system.f(data, system.advanceConfiguration(x, 0.5 * self.k1), u))
    np.copyto(self.k3, dt * system.f(data, system.advanceConfiguration(x, 0.5 * self.k2), u))
    np.copyto(self.k4, dt * system.f(data, system.advanceConfiguration(x, self.k3), u))
    np.copyto(self.sum_k, 1. / 6 * (self.k1 + 2. * self.k2 + 2. * self.k3 + self.k4))
    np.copyto(data.f, system.advanceConfiguration(x, self.sum_k))
    return data.f


def computeFlow(integrator, system, data, timeline, x, controls):
  N = len(timeline)
  x_flow = [None] * N
  x_flow[0] = x
  for k in range(len(timeline)-1):
    dt = timeline[k+1] - timeline[k]
    t = timeline[k]
    control = controls[k]
    x_flow[k+1] = integrator.integrate(system, data, t, x_flow[k], control, dt)

  return x_flow
