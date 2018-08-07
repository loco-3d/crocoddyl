import abc
import numpy as np
import math
from cddp.system import DynamicalSystem


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
  def computePerturbedConfiguration(self, x, index):
    """ Compute the perturbed configuration by perturbing its tangent space.

    In general, computing the perturbed configuration is done by using an
    integrator. However, this integrator depends on the manifold itself (e.g.
    SE(3) manifold). So, this integration rule depends on the particular
    diffeomorphism of our dynamical system. For instance, in a classical system,
    we might compute this quantity as x[index] += sqrt(eps), where eps is the
    machine epsilon; you can see an implementation in SpringMass system class.

    :param x: configuration state
    :param index: index (in the configuration state) for computing the
    perturbation
    """
    pass

  def computePerturbedControl(self, u, index):
    """ Compute the perturbed control.

    We assume that the control space lie in real coordinate space where we
    can apply classical calculus.
    :param u: control input
    :param index: index for computing the perturbation
    """
    u_pert = u.copy()
    u_pert[index] += self.sqrt_eps
    return u_pert

  def fx(self, data, x, u):
    """ Compute numerically the state derivative of the system dynamics and
    stores the result in data.

    :param data: dynamic system data
    :param x: configuration state
    :param u: control input
    """
    np.copyto(self.f_nom, self.f(data, x, u))
    for i in range(data.nv):
      x_pert = self.computePerturbedConfiguration(x, i)
      data.fx[:, i] = (self.f(data, x_pert, u).copy() - self.f_nom) / self.sqrt_eps
    np.copyto(data.f, self.f_nom)
    return data.fx

  def fu(self, data, x, u):
    """ Compute numerically the control derivative of the system dynamics and
    stores the result in data.

    :param data: dynamic system data
    :param x: configuration state
    :param u: control input
    """
    np.copyto(self.f_nom, self.f(data, x, u))
    for i in range(data.m):
      u_pert = self.computePerturbedControl(u, i)
      data.fu[:, i] = (self.f(data, x, u_pert).copy() - self.f_nom) / self.sqrt_eps
    np.copyto(data.f, self.f_nom)
    return data.fu