import abc
import numpy as np


class Discretizer(object):
  """ This abstract class declares the virtual method for any discretization
  method.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    pass
  
  @abc.abstractmethod
  def createData(self, nv):
    """ Create the data for the numerical integrator and discretizer.

    Due to the integrator data is needed internally, we create inside the
    implementation class. It aims is to create data for the integrator and
    sampler, and both depends on the dimension of the tangential space nv.
    :param nv: dimension of the tangent space of the configuration manifold
    """
    pass

  @abc.abstractmethod
  def __call__(self, system, data, x, u, dt):
    """ Convert the continuos, and linearized, dynamic model into discrete one.

    This abstract method allows us to define a discretization rule for the 
    linearized dynamic (dv = fx*dx + fu*du) which is compatible with its 
    integration scheme. Note that we assume a first-order Taylor linearization.
    :param system: system model
    :param data: system data
    :para x: state vector
    :param u: control vector
    :param dt: sampling period
    :returns: time-discrete state and control derivatives
    """
    pass


class GeometricDiscretizer(object):
  """ This abstract class declares the virtual method for any discretization
  method.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    pass
  
  @abc.abstractmethod
  def createData(self, nq, nv):
    """ Create the data for the numerical integrator and discretizer.

    Due to the integrator data is needed internally, we create inside the
    implementation class. It aims is to create data for the integrator and
    sampler, and both depends on the dimension of the tangential space nv.
    :param nq: dimension of the configuration space Q
    :param nv: dimension of the tangent space of the configuration manifold
    """
    pass


  @abc.abstractmethod
  def __call__(self, system, data, q, v, tau, dt):
    """ Convert the continuos, and linearized, dynamic model into discrete one.

    This abstract method allows us to define a discretization rule for the 
    linearized dynamic ([dv,da] = fx*[dq,dv] + fu*du) which is compatible with
    its integration scheme. Note that we assume a first-order Taylor
    linearization.
    :param system: system model
    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: torque input
    :param dt: sampling period
    :returns: time-discrete state and control derivatives
    """
    pass


class EulerDiscretizer(Discretizer):
  def createData(self, nv):
    """ Create the internal data of forward Euler discretizer.

    :param nv: dimension of the tangent space of the configuration manifold
    """
    # Data for discretization
    self._I = np.eye(nv)

  def __call__(self, system, data, x, u, dt):
    """ Convert the time-continuos dynamics into discrete one using forward
    Euler rule.

    :param system: system model
    :param data: system data
    :param x: state vector
    :param u: control vector
    :param dt: sampling period
    :returns: discrete time state and control derivatives
    """
    system.fx(data, x, u)
    system.fu(data, x, u)
    np.copyto(data.fx, self._I + data.fx * dt)
    np.copyto(data.fu, data.fu * dt)
    return data.fx, data.fu


class GeometricEulerDiscretizer(GeometricDiscretizer):
  def createData(self, nv):
    """ Create the internal data of forward Euler discretizer.

    :param nv: dimension of the tangent space of the configuration manifold
    """
    # Data for discretization
    self._I = np.eye(nv)

  def __call__(self, system, data, q, v, tau, dt):
    """ Convert the time-continuos dynamics into discrete one using forward
    Euler rule.

    :param system: system model
    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: control vector
    :param dt: sampling period
    :returns: discrete time state and control derivatives
    """
    system.g(data, q, v, tau)
    system.gq(data, q, v, tau)
    system.gv(data, q, v, tau)
    system.gtau(data, q, v, tau)
    np.copyto(data.gq, data.gq * dt)
    np.copyto(data.gv, self._I + data.gv * dt)
    np.copyto(data.gtau, data.gtau * dt)

    data.fx[:data.nv,:data.nv] = self._I
    data.fx[:data.nv,data.nv:] = self._I * dt
    data.fx[data.nv:,:data.nv] = data.gq
    data.fx[data.nv:,data.nv:] = data.gv
    data.fu[:data.nv,:] = np.zeros((data.nv, data.m))
    data.fu[data.nv:,:] = data.gtau
    return data.fx, data.fu


class RK4Discretizer(Discretizer):
  def createData(self, nv):
    # Data for discretization
    self._I = np.eye(nv)
    self.f1 = np.matrix(np.zeros((nv, nv)))
    self.f2 = np.matrix(np.zeros((nv, nv)))
    self.f3 = np.matrix(np.zeros((nv, nv)))
    self.f4 = np.matrix(np.zeros((nv, nv)))
    self.sum_f = np.matrix(np.zeros((nv, nv)))

  def __call__(self, system, data, x, u, dt):
    """ Convert the time-continuos dynamics into discrete one using
    fourth-order Runge-Kutta method.

    :param system: system model
    :param data: system data
    :param x: state vector
    :param u: control vector
    :param dt: sampling period
    :returns: discrete time state and control derivatives
    """
    # Computing four stages of RK4
    system.f(data, x, u)
    system.fx(data, x, u)
    system.fu(data, x, u)
    np.copyto(self.f1, dt * self._I)
    np.copyto(self.f2, dt * (self._I + 0.5 * data.fx * dt))
    np.copyto(self.f3, dt * (self._I + 0.5 * data.fx * self.f2))
    np.copyto(self.f4, dt * (self._I + data.fx * self.f3))
    np.copyto(self.sum_f, \
      1. / 6 * (self.f1 + 2. * self.f2 + 2. * self.f3 + self.f4))

    # Computing the discrete time state and control derivatives
    np.copyto(data.fx, self._I + self.sum_f * data.fx)
    np.copyto(data.fu, self.sum_f * data.fu)
    return data.fx, data.fu