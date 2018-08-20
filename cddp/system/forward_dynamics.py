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
  def __init__(self, urdf, path):
    """ Construct the robot forward dynamic model.

    :param urdf: URDF file
    :param path: package path
    """
    # Getting the Pinocchio model of the robot
    self.robot = se3.robot_wrapper.RobotWrapper(urdf, path)
    self.rmodel = self.robot.model
    self.rdata = self.robot.data

    # Initializing the dynamic model with numerical differentiation
    nq = self.robot.nq + self.robot.nv
    nv = self.robot.nv + self.robot.nv
    m = self.robot.nv
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
    q = x[:self.robot.nq]
    v = x[self.robot.nq:]
    se3.aba(self.rmodel, self.rdata, q, v, u)
    data.f[:self.robot.nq] = v
    data.f[self.robot.nq:] = self.rdata.ddq
    return data.f

  def advanceConfiguration(self, x, dx):
    """ Describe the operator that advance the configuration state.

    :param x: configuration state [joint configuration, joint velocity]
    :param dx: configuration state displacement
    :returns: the next configuration state
    """
    q = x[:self.robot.nq]
    dq = dx[:self.robot.nv]
    x[:self.robot.nq] = se3.integrate(self.rmodel, q, dq)
    x[self.robot.nq:] += dx[self.robot.nv:]
    return x


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
  def __init__(self, urdf, path):
    """ Construct the robot forward dynamics model.

    :param urdf: URDF file
    :param path: package path
    """
    # Getting the Pinocchio model of the robot
    self.robot = se3.robot_wrapper.RobotWrapper(urdf, path)
    self.rmodel = self.robot.model
    self.rdata = self.robot.data

    # Initializing the dynamic model with numerical differentiation
    nq = self.robot.nq
    nv = self.robot.nv
    m = self.robot.nv
    integrator = GeometricEulerIntegrator()
    discretizer = GeometricEulerDiscretizer()
    NumDiffGeometricDynamicalSystem.__init__(self, nq, nv, m, integrator, discretizer)

  def g(self, data, q, v, tau):
    """ Compute the forward dynamics through ABA and store it the result in
    data.

    :param data: geometric system data
    :param x: configuration state [joint configuration, joint velocity]
    :param u: control input
    """
    se3.aba(self.rmodel, self.rdata, q, v, tau)
    np.copyto(data.g, self.rdata.ddq)
    return data.g

  def advanceConfiguration(self, q, dq):
    """ Describe the operator that advance the configuration state.

    :param q: configuration point
    :param dq: configuration point displacement
    :returns: the next configuration point
    """
    return se3.integrate(self.rmodel, q, dq)