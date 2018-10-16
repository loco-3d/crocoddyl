from cddp.system import GeometricDynamicalSystem, NumDiffGeometricDynamicalSystem
from cddp.math import GeometricEulerIntegrator, GeometricEulerDiscretizer
import numpy as np
import pinocchio as se3


class NumDiffSparseConstrainedForwardDynamics(NumDiffGeometricDynamicalSystem):
  """ Sparse robot constrained forward dynamics with numerical computation of
  derivatives.

  The continuos evolution function (i.e. f(q,v,tau)=[v, g(q,v,tau)]) is defined
  by the current joint velocity and the forward dynamics g() for a given
  constrained contact set. Describing as geometrical system allows us to exploit
  the sparsity of the derivatives computation and to preserve the geometry of
  the Lie manifold thanks to a sympletic integration rule. The Jacobian are
  computed through numerical differentiation.
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

    self.contact_set = []
    self.contact_set.append(self._model.getFrameId("lf_foot"))
    # self.contact_set.append(self._model.getFrameId("lh_foot"))
    self.contact_set.append(self._model.getFrameId("rf_foot"))
    self.contact_set.append(self._model.getFrameId("rh_foot"))
    self.nc = 3
    self.J = np.matrix(np.zeros((self.nc * len(self.contact_set), self._model.nv)))
    self.gamma = np.matrix(np.zeros((self.nc * len(self.contact_set), 1)))

    # Internal data for operators
    self._q = np.matrix(np.zeros((self.nq,1)))
    self._v = np.matrix(np.zeros((2*self.nv,1)))

  def g(self, data, q, v, tau):
    """ Compute the forward dynamics given a constrained contact set and store
    it the result in data.

    :param data: geometric system data
    :param q: joint configuration
    :param v: joint velocity
    :param tau: torque input
    """
    # Updating the frame kinematics and Jacobians
    se3.forwardKinematics(self._model, self._data, q)
    se3.computeJointJacobians(self._model, self._data, q)
    se3.framesKinematics(self._model, self._data, q)


    zero_motion = se3.Motion(np.zeros((3,1)), np.zeros((3,1)))
    a_ref = 0.*self.gamma.copy()
    for k in range(len(self.contact_set)):
      # Getting the frame and its ID
      frame_id = self.contact_set[k]
      frame = self._model.frames[frame_id]
      
      # Computing the frame Jacobian in the local frame
      self.J[self.nc*k:self.nc*(k+1),:] = \
        se3.getFrameJacobian(self._model, self._data, frame_id,
                          se3.ReferenceFrame.LOCAL)[:self.nc,:]

      # Mapping the reference acceleration into the local frame
      oMf = self._data.oMi[frame.parent].act(frame.placement)
      a_ref[self.nc*k:self.nc*k+self.nc] = oMf.actInv(zero_motion).vector[:self.nc]

    # Computing the under-actuated torque command vector
    Stau = tau.copy()
    Stau[:6] *= 0.

    # print Stau.T
    # Stau *= 0.
    se3.forwardDynamics(self._model, self._data, q, v, Stau,
                        self.J, a_ref,  1e-8, True)#, 1e-12)
    # print self._data.lambda_c.T
    # print
    np.copyto(data.g, self._data.ddq)
    return data.g

  def advanceConfiguration(self, q, dq):
    """ Operator that advances the configuration state.

    :param q: configuration point
    :param dq: configuration point displacement
    :returns: the next configuration point
    """
    np.copyto(self._q, se3.integrate(self._model, q, dq))
    return self._q

  def differenceConfiguration(self, x_next, x_curr):
    """ Operator that differentiates the configuration state.

    :param x_next: next joint configuration and velocity [q_next, v_next]
    :param x_curr: current joint configuration and velocity [q_curr, v_curr]
    """
    q_next = x_next[:self.nq]
    q_curr = x_curr[:self.nq]
    self._v[:self.nv] = se3.difference(self._model, q_curr, q_next)
    self._v[self.nv:] = x_next[self.nq:] - x_curr[self.nq:]
    return self._v
