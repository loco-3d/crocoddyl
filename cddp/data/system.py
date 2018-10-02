import abc
import numpy as np


class SystemData(object):
  """ Data structure for the system dynamics.

  We only define the Jacobians of the dynamics, and not the Hessians of it,
  because our optimal controller by default uses the Gauss-Newton approximation.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, nq, nv, m):
    # Dimension of the phase state manifold and its tangent
    self.nx = nq
    self.ndx = nv

    # Configuration and its tangent and control dimensions
    self.nq = nq
    self.nv = nv
    self.m = m

    # Creating the data structure for the ODE and its derivative
    self.f = np.matrix(np.zeros((self.nx, 1)))
    self.fx = np.matrix(np.zeros((self.ndx, self.ndx)))
    self.fu = np.matrix(np.zeros((self.ndx, self.m)))


class GeometricSystemData(object):
  """ Data structure for the geometric system dynamics.

  We only define the Jacobians of the dynamics, and not the Hessians of it,
  because our optimal controller by default uses the Gauss-Newton approximation.
  """
  __metaclass__ = abc.ABCMeta

  def __init__(self, nq, nv, m):
    # Dimension of the phase state manifold and its tangent
    self.nx = nq + nv
    self.ndx = 2 * nv

    # Configuration and its tangent and control dimensions
    self.nq = nq
    self.nv = nv
    self.m = m

    # Creating the data structure for the ODE and its derivative
    self.f = np.matrix(np.zeros((self.nx, 1)))
    self.fx = np.matrix(np.zeros((self.ndx, self.ndx)))
    self.fu = np.matrix(np.zeros((self.ndx, self.m)))

    self.g = np.matrix(np.zeros((self.nv, 1)))
    self.gq = np.matrix(np.zeros((self.nv, self.nv)))
    self.gv = np.matrix(np.zeros((self.nv, self.nv)))
    self.gtau = np.matrix(np.zeros((self.nv, self.m)))