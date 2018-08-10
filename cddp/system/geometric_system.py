import abc
import math
import numpy as np
from cddp.system import DynamicalSystem
from cddp.utils import assertClass


class GeometricDynamicalSystem(DynamicalSystem):
  """ This abstract class declares virtual methods for defining the geometric
  system evolution and its derivatives.

  It allows us to define any kind of geometric mechanical system of the form
  v,a = f(q,v,tau). The function evolution f() is described as [v, g(q,v,tau)].
  The configuration point q, or some part of it, belongs to the Lie group; a
  differential manifold denoted as Q. Instead the vector fields (v,a) lie in
  the tangent space TqQ of a given configuration point q. The control (input)
  vector tau allows the system to evolves toward a desired configuration point.
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
    assertClass(integrator, 'GeometricIntegrator')
    assertClass(discretizer, 'GeometricDiscretizer')

    self.nq = nq
    self.nv = nv
    self.m = m
    self.integrator = integrator
    self.discretizer = discretizer

    # Creates internally the integrator and discretizer data
    self.integrator.createData(nq, nv)
    self.discretizer.createData(nv)

  def createData(self):
    """ Create the geometric system data.
    """
    from cddp.data.system import GeometricSystemData
    return GeometricSystemData(self.nq, self.nv, self.m)

  def stepForward(self, data, x, u, dt):
    # Integrate the time-continuos dynamics in order to get the next state
    # value
    q = x[:self.nq]
    v = x[self.nq:]
    return self.integrator(self, data, q, v, u, dt)

  def computeDerivatives(self, data, x, u, dt):
    # Computing the time-continuos linearized system, i.e.
    # [dq,dv]^T = fqv*[q,v]^T + [0,fu]*du, and converting it into discrete one
    q = x[:self.nq]
    v = x[self.nq:]
    self.discretizer(self, data, q, v, u, dt)
    return data.fx, data.fu

  def f(self, data, x, u):
    """ Evaluate the evolution function of a geometric system and stores the
    result in data.

    :param data: dynamics data
    :param x: state vector (q,v)
    :param u: control input
    :returns: state velocity
    """
    # Getting the configuration point and its generalized velocity
    q = x[:self.nq]
    v = x[self.nq:]

    # Computing the system evolution
    data.f[:self.nv] = v
    data.f[self.nv:] = self.g(data, q, v, u)
    return data.f

  def fx(self, data, x, u):
    """ Evaluate the geometric system Jacobian w.r.t. the state and stores
    the result in data.

    :param data: dynamics data
    :param x: state vector (q,v)
    :param u: control input
    :returns: geometric system Jacobian w.r.t the state
    """
    # Getting the configuration point and its generalized velocity
    q = x[:self.nq]
    v = x[self.nq:]

    # Computing the Jacobian w.r.t. the configuration point (fq) and its
    # tangent space (fv)
    data.fx[:self.nv,:self.nv] = np.zeros((self.nv, self.nv))
    data.fx[:self.nv,self.nv:] = np.eye(self.nv)
    data.fx[self.nv:,:self.nv] = self.gq(data, q, v, u)
    data.fx[self.nv:,self.nv:] = self.gv(data, q, v, u)
    return data.fx

  def fu(self, data, x, u):
    # Getting the configuration point and its generalized velocity
    q = x[:self.nq]
    v = x[self.nq:]

    # Computing the Jacobian w.r.t torque input
    data.fu[:self.nv,:] = np.zeros((self.nv, self.m))
    data.fu[self.nv:,:] = self.gtau(data, q, v, u)
    return data.fu

  @abc.abstractmethod
  def advanceGeometric(self, q, dq):
    """ Operator that advance the configuration state

    :param q: configuration point
    :param dq: displacement in tangent space of configuration manifold
    :returns: next configuration point
    """
    pass

  @abc.abstractmethod
  def g(self, data, q, v, tau):
    """ Evaluate the evolution function and stores the result in data.

    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: torque input
    :returns: generalized acceleration
    """
    pass

  @abc.abstractmethod
  def gq(self, data, q, v, tau):
    """ Evaluate the system Jacobian w.r.t. the configuration point and store
    the result in data.

    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: torque input
    :returns: system Jacobian w.r.t the configuration point
    """
    pass
  
  @abc.abstractmethod
  def gv(self, data, q, v, tau):
    """ Evaluate the system Jacobian w.r.t. the generalized velocity and stores
    the result in data.

    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: torque input
    :returns: system Jacobian w.r.t the generalized velocity
    """
    pass

  @abc.abstractmethod
  def gtau(self, data, q, v, tau):
    """ Evaluate the system Jacobian w.r.t. the torque input and stores
    the result in data.

    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: torque input
    :returns: system Jacobian w.r.t the torque input
    """
    pass



class NumDiffGeometricDynamicalSystem(GeometricDynamicalSystem):
  """ This abstract class declares virtual methods for defining the geometric
  system evolution where its derivatives are computed numerically.

  This class uses numerical differentiation for computing the state and control
  derivatives of a dynamic model.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, nq, nv, m, integrator, discretizer):
    GeometricDynamicalSystem.__init__(self, nq, nv, m, integrator, discretizer)
    self.sqrt_eps = math.sqrt(np.finfo(float).eps)
    self.g_nom = np.matrix(np.zeros((nv, 1)))

  @abc.abstractmethod
  def advanceGeometric(self, q, dq):
    """ Operator that advance the configuration state

    :param q: configuration point
    :param dq: displacement in tangent space of configuration manifold
    :returns: next configuration point
    """
    pass

  @abc.abstractmethod
  def g(self, data, q, v, tau):
    """ Evaluate the evolution function and stores the result in data.

    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: torque input
    :returns: generalized acceleration
    """
    pass

  def gq(self, data, q, v, tau):
    """ Compute numerically the Jacobian of the system w.r.t. the configuration
    point and stores the result in data.

    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: torque input
    :returns: system Jacobian w.r.t. the configuration point
    """
    np.copyto(self.g_nom, self.g(data, q, v, tau))
    for i in range(data.nv):
      q_pert = q.copy()
      v_pert = np.zeros((self.robot.nv, 1))
      v_pert[i] += self.sqrt_eps
      q_pert[:self.robot.nq] = self.advanceGeometric(q, v_pert)
      data.gq[:, i] = \
        (self.g(data, q_pert, v, tau).copy() - self.g_nom) / self.sqrt_eps
    np.copyto(data.g, self.g_nom)
    return data.gq

  def gv(self, data, q, v, tau):
    """ Compute numerically the Jacobian of the system w.r.t. the generalized
    velocity and stores the result in data.

    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: torque input
    :returns: system Jacobian w.r.t. the generalized velocity
    """
    np.copyto(self.g_nom, self.g(data, q, v, tau))
    for i in range(data.nv):
      v_pert = v.copy()
      v_pert[i] += self.sqrt_eps
      data.gv[:, i] = \
        (self.g(data, q, v_pert, tau).copy() - self.g_nom) / self.sqrt_eps
    np.copyto(data.g, self.g_nom)
    return data.gv

  def gtau(self, data, q, v, tau):
    """ Compute numerically the Jacobian of the system w.r.t. the torque input
    and stores the result in data.

    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: torque input
    :returns: system Jacobian w.r.t. the torque input
    """
    np.copyto(self.g_nom, self.g(data, q, v, tau))
    for i in range(data.nv):
      tau_pert = tau.copy()
      tau_pert[i] += self.sqrt_eps
      data.gtau[:, i] = \
        (self.g(data, q, v, tau_pert).copy() - self.g_nom) / self.sqrt_eps
    np.copyto(data.g, self.g_nom)
    return data.gtau