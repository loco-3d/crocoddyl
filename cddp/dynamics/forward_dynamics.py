from cddp.dynamics.dynamics import DynamicsData
from cddp.dynamics.dynamics import DynamicsModel
import pinocchio as se3
import numpy as np


class ForwardDynamicsData(DynamicsData):
  def __init__(self, dynamicModel, t, dt):
    DynamicsData.__init__(self, dynamicModel, t, dt)

    # Pinocchio data
    self.pinocchio = dynamicModel.pinocchio.createData()


class ForwardDynamics(DynamicsModel):
  """ Forward dynamics computed by the Articulated Body Algorithm (ABA).

  The ABA algorithm computes the forward dynamics for unconstrained rigid body
  system; it cannot be modeled contact interations. The continuos evolution
  function (i.e. f(q,v,tau)=[v, a(q,v,tau)]) is defined by the current joint
  velocity and the forward dynamics a() which is computed using the ABA. 
  Describing as geometrical system allows us to exploit the sparsity of the
  derivatives computation and to preserve the geometry of the SE3 manifold
  thanks to a sympletic integration rule.
  """
  def __init__(self, integrator, discretizer, pinocchioModel):
    DynamicsModel.__init__(self, integrator, discretizer,
                           pinocchioModel.nq,
                           pinocchioModel.nv,
                           pinocchioModel.nv)
    self.pinocchio = pinocchioModel

  def createData(self, t, dt):
    return ForwardDynamicsData(self, t, dt)

  def updateTerms(self, dynamicsData, x):
    # Compute all terms
    #TODO: Try to reduce calculations in forward pass, and move them to backward pass
    se3.computeAllTerms(self.pinocchio, dynamicsData.pinocchio,
                        x[:self.nq()], x[self.nq():])
    se3.updateFramePlacements(self.pinocchio,
                              dynamicsData.pinocchio)

  def updateDynamics(self, dynamicsData, x, u):
    # Update all terms
    self.updateTerms(dynamicsData, x)

    # Running ABA algorithm
    se3.aba(self.pinocchio, dynamicsData.pinocchio,
            x[:self.nq()], x[self.nq():], u)

    # Updating the system acceleration
    np.copyto(dynamicsData.a, dynamicsData.pinocchio.ddq)

  def updateLinearAppr(self, dynamicsData, x, u):
    se3.computeABADerivatives(self.pinocchio, dynamicsData.pinocchio,
                              x[:self.nq()], x[self.nq():], u)

    # Updating the system derivatives
    np.copyto(dynamicsData.aq, dynamicsData.pinocchio.ddq_dq)
    np.copyto(dynamicsData.av, dynamicsData.pinocchio.ddq_dv)
    np.copyto(dynamicsData.au, dynamicsData.pinocchio.Minv)

  def integrateConfiguration(self, q, dq):
    return se3.integrate(self.pinocchio, q, dq)

  def differenceConfiguration(self, q0, q1):
    return se3.difference(self.pinocchio, q0, q1)