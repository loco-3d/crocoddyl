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
  def __call__(self, model, data, x, u, dt):
    """ Integrate the system dynamics.

    This abstract method allows us to define integration rules of our ODE. It
    uses the model and data classes which defines the evolution function and
    its data, respectively. For more information about the model and data see
    the Dynamics class and and its data structure.
    :param model: system model
    :param data: system data
    :para x: state vector
    :param u: control vector
    :param dt: integration step
    :returns: next state vector
    """
    pass


class EulerIntegrator(Integrator):
  def createData(self, nv):
    """ Create the internal data of forward Euler integrator.

    :param nv: dimension of the tangent space of the configuration manifold
    """
    # Data for integration
    self.x_next = np.matrix(np.zeros((nv, 1)))

  def __call__(self, model, data, x, u, dt):
    """ Integrate the system dynamics using the forward Euler scheme.

    :param model: system model
    :param data: system data
    :para x: state vector
    :param u: control vector
    :param dt: sampling period
    :returns: next state vector
    """
    np.copyto(self.x_next, x + dt * model.f(data, x, u))
    return self.x_next


class RK4Integrator(Integrator):
  def createData(self, nv):
    # Data for integration
    self.x_next = np.matrix(np.zeros((nv, 1)))
    self.k1 = np.matrix(np.zeros((nv, 1)))
    self.k2 = np.matrix(np.zeros((nv, 1)))
    self.k3 = np.matrix(np.zeros((nv, 1)))
    self.k4 = np.matrix(np.zeros((nv, 1)))

  def __call__(self, model, data, x, u, dt):
    """ Integrate the system dynamics using the fourth-order Runge-Kutta method.

    :param model: system model
    :param data: system data
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
