import abc


class Integrator(object):
  """ This abstract class declared the virtual method for any integrator.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self):
    pass

  @abc.abstractmethod
  def integrate(self, model, data, x, u, dt): pass


class EulerIntegrator(Integrator):
  """ Integrates the function using the forward Euler method
  """
  @staticmethod
  def integrate(model, data, x, u, dt):
    return x + model.f(data, x, u) * dt


class RK4Integrator(Integrator):
  """ Integrates the function using the fourth-order Runge-Kutta method
  """
  @staticmethod
  def integrate(model, data, x, u, dt):
    k1 = model.f(data, x, u)

    x2 = x + dt / 2. * k1
    k2 = model.f(data, x2, u)

    x3 = x + dt / 2. * k2
    k3 = model.f(data, x3, u)

    x4 = x + dt * k3
    k4 = model.f(data, x4, u)
    return x + dt / 6. * (k1 + 2. * k2 + 2. * k3 + k4)


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
