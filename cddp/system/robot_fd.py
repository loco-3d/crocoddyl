from cddp.system import DynamicalSystem, NumDiffDynamicalSystem
from cddp.system import GeometricDynamicalSystem, NumDiffGeometricDynamicalSystem
from cddp.math import EulerIntegrator, EulerDiscretizer
from cddp.math import GeometricEulerIntegrator, GeometricEulerDiscretizer
import numpy as np
import pinocchio as se3


class NumDiffRobotFD(NumDiffDynamicalSystem):
  def __init__(self, urdf, path):
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
    q = x[:self.robot.nq]
    v = x[self.robot.nq:]
    se3.aba(self.rmodel, self.rdata, q, v, u)
    data.f[:self.robot.nq] = v
    data.f[self.robot.nq:] = self.rdata.ddq
    return data.f

  def advanceConfiguration(self, x, dx):
    q = x[:self.robot.nq]
    dq = dx[:self.robot.nv]
    x[:self.robot.nq] = se3.integrate(self.rmodel, q, dq)
    x[self.robot.nq:] += dx[self.robot.nv:]
    return x


class NumDiffSparseRobotFD(NumDiffGeometricDynamicalSystem):
  def __init__(self, urdf, path):
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
    se3.aba(self.rmodel, self.rdata, q, v, tau)
    np.copyto(data.g, self.rdata.ddq)
    return data.g

  def advanceConfiguration(self, q, dq):
    return se3.integrate(self.rmodel, q, dq)