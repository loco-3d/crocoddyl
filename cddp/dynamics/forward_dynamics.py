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

  def createData(dynamicsModel, t, dt):
    return ForwardDynamicsData(dynamicsModel, t, dt)

  def updateTerms(dynamicsModel, dynamicsData):
    # Compute all terms
    #TODO: Try to reduce calculations in forward pass, and move them to backward pass
    se3.computeAllTerms(dynamicsModel.pinocchio,
                        dynamicsData.pinocchio,
                        dynamicsData.x[:dynamicsModel.nq()],
                        dynamicsData.x[dynamicsModel.nq():])
    se3.updateFramePlacements(dynamicsModel.pinocchio,
                              dynamicsData.pinocchio)

  def updateDynamics(dynamicsModel, dynamicsData):
    # Running ABA algorithm
    se3.aba(dynamicsModel.pinocchio,
            dynamicsData.pinocchio,
            dynamicsData.x[:dynamicsModel.nq()],
            dynamicsData.x[dynamicsModel.nq():],
            dynamicsData.u)

    # Updating the system acceleration
    np.copyto(dynamicsData.a, dynamicsData.pinocchio.ddq)
    pass

  def updateLinearAppr(dynamicsModel, dynamicsData):
    se3.computeABADerivatives(dynamicsModel.pinocchio,
                              dynamicsData.pinocchio,
                              dynamicsData.x[:dynamicsModel.nq()],
                              dynamicsData.x[dynamicsModel.nq():],
                              dynamicsData.u)
    np.copyto(dynamicsData.aq, dynamicsData.pinocchio.ddq_dq)
    np.copyto(dynamicsData.av, dynamicsData.pinocchio.ddq_dv)
    np.copyto(dynamicsData.au, dynamicsData.pinocchio.Minv)
    pass

  def integrateConfiguration(dynamicsModel, dynamicsData, q, dq):
    return se3.integrate(dynamicsModel.pinocchio, q, dq)

  def differenceConfiguration(dynamicsModel, dynamicsData, q0, q1):
    return se3.difference(dynamicsModel.pinocchio, q0, q1)