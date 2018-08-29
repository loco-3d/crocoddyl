from cddp.system import DynamicalSystem, NumDiffDynamicalSystem
from cddp.system import GeometricDynamicalSystem, NumDiffGeometricDynamicalSystem
from cddp.math import EulerIntegrator, EulerDiscretizer
from cddp.math import GeometricEulerIntegrator, GeometricEulerDiscretizer
import numpy as np
import pinocchio as se3


class NumDiffForwardDynamics(NumDiffDynamicalSystem):
  """ Robot forward dynamics with numerical computation of derivatives.

  The continuos evolution function (i.e. f(x, u)) is defined by the robot
  forward dynamics which is computed using the Articulated Body Algorithm (ABA).
  The ABA computes the forward dynamics of kinematic tree for unconstrained
  rigid body system. Indeed it does not model contact interations. The Jacobian
  are computed through numerical differentiation.
  """
  def __init__(self, model):
    """ Construct the robot forward dynamic model.

    :param model: Pinocchio model
    """
    # Getting the Pinocchio model of the robot
    self._model = model
    self._data = self._model.createData()

    # Initializing the dynamic model with numerical differentiation
    nq = self._model.nq + self._model.nv
    nv = self._model.nv + self._model.nv
    m = self._model.nv
    integrator = EulerIntegrator()
    discretizer = EulerDiscretizer()
    NumDiffDynamicalSystem.__init__(self, nq, nv, m, integrator, discretizer)

  def f(self, data, x, u):
    """ Compute the forward dynamics through ABA and store it the result in
    data.

    :param data: system data
    :param x: configuration state [joint configuration, joint velocity]
    :param u: control input
    """
    q = x[:self._model.nq]
    v = x[self._model.nq:]
    se3.aba(self._model, self._data, q, v, u)
    data.f[:self._model.nq] = v
    data.f[self._model.nq:] = self._data.ddq
    return data.f

  def advanceConfiguration(self, x, dx):
    """ Operator that advances the configuration state.

    :param x: configuration state [joint configuration, joint velocity]
    :param dx: configuration state displacement
    :returns: the next configuration state
    """
    q = x[:self._model.nq]
    dq = dx[:self._model.nv]
    x[:self._model.nq] = se3.integrate(self._model, q, dq)
    x[self._model.nq:] += dx[self._model.nv:]
    return x

  def differenceConfiguration(self, x_next, x_curr):
    """ Operator that differentiates the configuration state.

    :param x_next: next configuration state [joint configuration, joint velocity]
    :param x_curr: current configuration state [joint configuration, joint velocity]
    """
    q_next = x_next[:self._model.nq]
    q_curr = x_curr[:self._model.nq]
    dq = se3.difference(self._model, q_curr, q_next)
    dv = x_next[self._model.nq:] - x_curr[self._model.nq:]
    return np.vstack([dq, dv])


class NumDiffSparseForwardDynamics(NumDiffGeometricDynamicalSystem):
  """ Sparse robot forward dynamics with numerical computation of derivatives.

  The continuos evolution function (i.e. f(q,v,tau)=[v, g(q,v,tau)]) is defined
  by the current joint velocity and the forward dynamics g() which is computed
  using the Articulated Body Algorithm (ABA). Describing as geometrical system
  allows us to exploit the sparsity of the derivatives computation and to
  preserve the geometry of the Lie manifold thanks to a sympletic integration
  rule. Note that ABA computes the forward dynamics for unconstrained rigid body
  system; it cannot be modeled contact interations. The Jacobian are computed
  through numerical differentiation.
  """
  def __init__(self, model):
    """ Construct the robot forward dynamics model.

    :param model: Pinocchio model
    """
    # Getting the Pinocchio model of the robot
    self._model = model
    self._data = self._model.createData()

    # Initializing the dynamic model with numerical differentiation
    nq = self._model.nq
    nv = self._model.nv
    m = self._model.nv
    integrator = GeometricEulerIntegrator()
    discretizer = GeometricEulerDiscretizer()
    NumDiffGeometricDynamicalSystem.__init__(self, nq, nv, m, integrator, discretizer)

  def g(self, data, q, v, tau):
    """ Compute the forward dynamics through ABA and store it the result in
    data.

    :param data: geometric system data
    :param q: joint configuration
    :param v: joint velocity
    :param tau: torque input
    """
    se3.aba(self._model, self._data, q, v, tau)
    np.copyto(data.g, self._data.ddq)
    return data.g

  def advanceConfiguration(self, q, dq):
    """ Operator that advances the configuration state.

    :param q: joint configuration
    :param dq: joint configuration displacement
    :returns: the next configuration point
    """
    return se3.integrate(self._model, q, dq)

  def differenceConfiguration(self, x_next, x_curr):
    """ Operator that differentiates the configuration state.

    :param x_next: next joint configuration and velocity [q_next, v_next]
    :param x_curr: current joint configuration and velocity [q_curr, v_curr]
    """
    q_next = x_next[:self.nq]
    q_curr = x_curr[:self.nq]
    dq = se3.difference(self._model, q_curr, q_next)
    dv = x_next[self.nq:] - x_curr[self.nq:]
    return np.vstack([dq, dv])


class SparseForwardDynamics(GeometricDynamicalSystem):
  """ Sparse robot forward dynamics with analytical derivatives.

  The continuos evolution function (i.e. f(q,v,tau)=[v, g(q,v,tau)]) is defined
  by the current joint velocity and the forward dynamics g() which is computed
  using the Articulated Body Algorithm (ABA). Describing as geometrical system
  allows us to exploit the sparsity of the derivatives computation and to
  preserve the geometry of the Lie manifold thanks to a sympletic integration
  rule. Note that ABA computes the forward dynamics for unconstrained rigid body
  system; it cannot be modeled contact interations.
  """
  def __init__(self, model):
    """ Construct the robot forward dynamics model.

    :param model: Pinocchio model
    """
    # Getting the Pinocchio model of the robot
    self._model = model
    self._data = self._model.createData()

    # Initializing the dynamic model with numerical differentiation
    nq = self._model.nq
    nv = self._model.nv
    m = self._model.nv
    integrator = GeometricEulerIntegrator()
    discretizer = GeometricEulerDiscretizer()
    GeometricDynamicalSystem.__init__(self, nq, nv, m, integrator, discretizer)

  def g(self, data, q, v, tau):
    """ Compute the ABA and its derivatives and store the ABA result in data.

    :param data: geometric system data
    :param q: joint configuration
    :param v: joint velocity
    :param tau: torque input
    """
    se3.computeABADerivatives(self._model, self._data, q, v, tau)
    np.copyto(data.g, self._data.ddq)
    return data.g

  def gq(self, data, q, v, tau):
    """ Compute the ABA derivatives and and store the result in data.

    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: torque input
    :returns: system Jacobian w.r.t the configuration point
    """
    np.copyto(data.gq, self._data.ddq_dq)
    return data.gq

  def gv(self, data, q, v, tau):
    """ Store the Jacobian w.r.t. the generalized velocity in data which it was
    previously computed by calling g() function

    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: torque input
    :returns: system Jacobian w.r.t the generalized velocity
    """
    np.copyto(data.gv, self._data.ddq_dv)

  def gtau(self, data, q, v, tau):
    """ Store the Jacobian w.r.t. the torque input in data which it was
    previously computed by calling g() function.

    :param data: system data
    :param q: configuration point
    :param v: generalized velocity
    :param tau: torque input
    :returns: system Jacobian w.r.t the torque input
    """
    np.copyto(data.gtau, self._data.Minv)

  def advanceConfiguration(self, q, dq):
    """ Operator that advances the configuration state.

    :param q: joint configuration
    :param dq: joint configuration displacement
    :returns: the next configuration point
    """
    return se3.integrate(self._model, q, dq)

  def differenceConfiguration(self, x_next, x_curr):
    """ Operator that differentiates the configuration state.

    :param x_next: next joint configuration and velocity [q_next, v_next]
    :param x_curr: current joint configuration and velocity [q_curr, v_curr]
    """
    q_next = x_next[:self._model.nq]
    q_curr = x_curr[:self._model.nq]
    dq = se3.difference(self._model, q_curr, q_next)
    dv = x_next[self._model.nq:] - x_curr[self._model.nq:]
    return np.vstack([dq, dv])