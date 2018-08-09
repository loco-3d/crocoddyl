import abc
import numpy as np

class Integrator(object):
  """ This abstract class declares the virtual method for any integrator.
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
  def integrate(self, model, data, x, u, dt):
    """ Integrate the system dynamics.

    This abstract method allows us to define integration rules of our ODE. It
    uses the model and data classes which defines the evolution function and
    its data, respectively. For more information about the model and data see
    the Dynamics class and and its data structure.
    :param model: dynamic model
    :param data: dynamic model data
    :para x: state vector
    :param u: control vector
    :param dt: integration step
    :returns: next state vector
    """
    pass
  
  @abc.abstractmethod
  def discretization(self, model, data, x, u, dt):
    """ Convert the continuos, and linearized, dynamic model into discrete one.

    This abstract method allows us to define a discretization rule for the 
    linearized dynamic (dv = fx*dx + fu*du) which is compatible withe its 
    integration scheme. Note that we assume a first-order Taylor linearization.
    :param model: dynamic model
    :param data: dynamic model data
    :para x: state vector
    :param u: control vector
    :param dt: sampling period
    :returns: time-discrete state and control derivatives
    """
    pass


class EulerIntegrator(Integrator):
  def createData(self, nv):
    """ Create the internal data of forward Euler sampler.

    :param nv: dimension of the tangent space of the configuration manifold
    """
    # Data for integration
    self.x_next = np.matrix(np.zeros((nv, 1)))

    # Data for discretization
    self._I = np.eye(nv)

  def integrate(self, model, data, x, u, dt):
    """ Integrate the system dynamics using the forward Euler scheme.

    :param model: dynamic model
    :param data: dynamic model data
    :para x: state vector
    :param u: control vector
    :param dt: sampling period
    :returns: next state vector
    """
    np.copyto(self.x_next, x + dt * model.f(data, x, u))
    return self.x_next

  def discretization(self, model, data, x, u, dt):
    """ Convert the time-continuos dynamics into discrete one using forward
    Euler rule.

    :param model: dynamic model
    :param data: dynamic model data
    :para x: state vector
    :param u: control vector
    :param dt: sampling period
    :returns: discrete time state and control derivatives
    """
    model.fx(data, x, u)
    model.fu(data, x, u)
    np.copyto(data.fx, self._I + data.fx * dt)
    np.copyto(data.fu, data.fu * dt)
    return data.fx, data.fu


class RK4Integrator(Integrator):
  def createData(self, nv):
    # Data for integration
    self.x_next = np.matrix(np.zeros((nv, 1)))
    self.k1 = np.matrix(np.zeros((nv, 1)))
    self.k2 = np.matrix(np.zeros((nv, 1)))
    self.k3 = np.matrix(np.zeros((nv, 1)))
    self.k4 = np.matrix(np.zeros((nv, 1)))

    # Data for discretization
    self._I = np.eye(nv)
    self.f1 = np.matrix(np.zeros((nv, nv)))
    self.f2 = np.matrix(np.zeros((nv, nv)))
    self.f3 = np.matrix(np.zeros((nv, nv)))
    self.f4 = np.matrix(np.zeros((nv, nv)))
    self.sum_f = np.matrix(np.zeros((nv, nv)))

  def integrate(self, model, data, x, u, dt):
    """ Integrate the system dynamics using the fourth-order Runge-Kutta method.

    :param model: dynamic model
    :param data: dynamic model data
    :para x: state vector
    :param u: control vector
    :param dt: sampling period
    :returns: next state vector
    """
    np.copyto(self.k1, dt * model.f(data, x, u))
    np.copyto(self.k2, dt * model.f(data, x + 0.5 * self.k1, u))
    np.copyto(self.k3, dt * model.f(data, x + 0.5 * self.k2, u))
    np.copyto(self.k4, dt * model.f(data, x + self.k3, u))
    np.copyto(self.x_next, x + \
      1. / 6 * (self.k1 + 2. * self.k2 + 2. * self.k3 + self.k4))
    return self.x_next

  def discretization(self, model, data, x, u, dt):
    """ Convert the time-continuos dynamics into discrete one using
    fourth-order Runge-Kutta method.

    :param model: dynamic model
    :param data: dynamic model data
    :para x: state vector
    :param u: control vector
    :param dt: sampling period
    :returns: discrete time state and control derivatives
    """
    # Computing four stages of RK4
    model.fx(data, x, u)
    model.fu(data, x, u)
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

def computeFlow(integrator, model, data, timeline, x, controls):
  N = len(timeline)
  x_flow = [None] * N
  x_flow[0] = x
  for k in range(len(timeline)-1):
    dt = timeline[k+1] - timeline[k]
    t = timeline[k]
    control = controls[k]
    x_flow[k+1] = integrator.integrate(model, data, t, x_flow[k], control, dt)

  return x_flow
