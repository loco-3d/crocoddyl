import abc
from ode import ODEBase


class DynamicModel(object):
  """ This abstract class declares virtual methods for defining the system
  evolution and its derivatives.

  It allows us to define any kind of smooth system dynamics of the form
  v = f(x,u), where nq, nv and m are the dimension of the configuration
  manifold x, its tangent space v and control u vectors, respectively.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, nq, nv, m):
    """ Constructs the dynamics model.

    :param nq: dimension of the configuration manifold
    :param nv: dimension of the tangent space of the configuration manifold
    :param m: dimension of the control space
    """
    self.nq = nq
    self.nv = nv
    self.m = m

  def createData(self):
    """ Creates the system dynamics data.
    """
    from data import DynamicsData
    return DynamicsData(self.nq, self.nv, self.m)

  def computeAllTerms(self, data, x, u):
    """ Computes the evolution function and its derivatives.
    """
    return self.f(data, x, u), self.fx(data, x, u), self.fu(data, x, u)

  @abc.abstractmethod
  def f(self, data, x, u):
    """ Evaluates the evolution function and stores the result in data.

    :param data: dynamics data
    :param x: system configuration
    :param u: control input
    :returns: state variation
    """
    pass

  @abc.abstractmethod
  def fx(self, data, x, u):
    """ Evaluates the dynamics Jacobian w.r.t. the state and stores the
    result in data.

    :param data: dynamics data
    :param x: system configuration
    :param u: control input
    :return: Jacobian of the state variation w.r.t the state
    """
    pass

  @abc.abstractmethod
  def fu(self, data, x, u):
    """ Evaluates the dynamics Jacobian w.r.t. the control and stores the
    result in data.

    :param data: dynamics data
    :param x: system configuration
    :param u: control input
    :returns: Jacobian of the state variation w.r.t the control
    """
    pass

  def stateDifference(self, xf, x0):
    """ Returns the state different between xf and x0 (i.e. xf - x0).

    :param xf: system configuration
    :param x0: system configuration
    """
    return xf - x0

  def getConfigurationDimension(self):
    """ Returns the configuration manifold dimension.

    :returns dimension of configuration manifold
    """
    return self.nq

  def getTangentDimension(self):
    """ Returns the tangent manifold dimension.

    :returns dimension of tangent space of the configuration manifold
    """
    return self.nv

  def getControlDimension(self):
    """ Returns the control dimension.

    :returns dimension of the control space
    """
    return self.m

  # @abc.abstractmethod
  # def fxx(self, data, x, u):
  #   """
  #   Eval the hessian of the dynamics with respect to the state
  #   :param x:
  #   :param u:
  #   :param data:
  #   :return:
  #   """
  #   pass

  # @abc.abstractmethod
  # def fxu(self, data, x, u):
  #   pass

  # @abc.abstractmethod
  # def fuu(self, data, x, u):
  #   pass


import numpy as np
import math
class NumDiffDynamicModel(DynamicModel):
  """ Computes the dynamic model evolution and its derivatives computation
  through numerical differentiation.

  This class uses numerical differentiation for computing the state and control
  derivatives of a dynamic model.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, nq, nv, m):
    """ Constructs the dynamics model.

    :param nq: dimension of the configuration manifold
    :param nv: dimension of the tangent space of the configuration manifold
    :param m: dimension of the control space
    """
    DynamicModel.__init__(self, nq, nv, m)
    self.sqrt_eps = math.sqrt(np.finfo(float).eps)
    self.f_nom = np.matrix(np.zeros((nv, 1)))

  @abc.abstractmethod
  def computePerturbedConfiguration(self, x, index):
    """ Computes the perturbed configuration by perturbing its tangent space.

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
    """ Computes the perturbed control.

    We assume that the control space lie in real coordinate space where we
    can apply classical calculus.
    :param u: control input
    :param index: index for computing the perturbation
    """
    u_pert = u.copy()
    u_pert[index] += self.sqrt_eps
    return u_pert

  def fx(self, data, x, u):
    """ Computes numerically the state derivative of the system dynamics and
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
    """ Computes numerically the control derivative of the system dynamics and
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