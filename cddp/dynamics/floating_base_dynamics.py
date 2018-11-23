from cddp.dynamics.dynamics import DynamicsModel
from cddp.dynamics.dynamics import DynamicsData
from cddp.utils import EPS
import pinocchio as se3
import numpy as np


class FloatingBaseMultibodyDynamicsData(DynamicsData):
  def __init__(self, dynamicsModel, t, dt):
    DynamicsData.__init__(self, dynamicsModel, t, dt)

    # Pinocchio data
    self.pinocchio = dynamicsModel.pinocchio.createData()

    # Constrained dynamics data (holonomic contacts)
    self.dimConstraint = \
      dynamicsModel.contactInfo.nc * dynamicsModel.contactInfo.dim(t)
    self.gq = np.zeros((self.dimConstraint, dynamicsModel.nv()))
    self.gv = np.zeros((self.dimConstraint, dynamicsModel.nv()))
    self.gu = np.zeros((self.dimConstraint, dynamicsModel.nu()))

    self.contactJ = np.zeros((self.dimConstraint, dynamicsModel.nv()))
    self.gamma = np.zeros((self.dimConstraint, 1))
    self._contactFrameIndices = dynamicsModel.contactInfo(t)
    self.MJtJc = np.zeros((dynamicsModel.nv() + self.dimConstraint,
                           dynamicsModel.nv() + self.dimConstraint))
    self.MJtJc_inv = np.zeros((dynamicsModel.nv() + self.dimConstraint,
                               dynamicsModel.nv() + self.dimConstraint))
    self.MJtJc_inv_L = np.zeros((dynamicsModel.nv() + self.dimConstraint,
                                 dynamicsModel.nv() + self.dimConstraint))

    # NumDiff data
    #TODO: remove these when replacing with analytical derivatives
    self.h = np.sqrt(EPS)
    self.q_pert = np.zeros((dynamicsModel.nq(), 1))
    self.v_pert = np.zeros((dynamicsModel.nv(), 1))


class FloatingBaseMultibodyDynamics(DynamicsModel):
  def __init__(self, integrator, discretizer, pinocchioModel, contactInfo):
    DynamicsModel.__init__(self, integrator, discretizer,
                           pinocchioModel.nq,
                           pinocchioModel.nv,
                           pinocchioModel.nv - 6)
    self.pinocchio = pinocchioModel
    self.contactInfo = contactInfo

  def createData(dynamicsModel, t, dt):
    return FloatingBaseMultibodyDynamicsData(dynamicsModel, t, dt)

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
    # Computing the constrained forward dynamics
    dynamicsModel.computeDynamics(dynamicsData,
                                  dynamicsData.x[:dynamicsModel.nq()],
                                  dynamicsData.x[dynamicsModel.nq():],
                                  np.vstack([np.zeros((6,1)), dynamicsData.u]))

    # Updating the system acceleration
    np.copyto(dynamicsData.a, dynamicsData.pinocchio.ddq)

  def updateLinearAppr(dynamicsModel, dynamicsData):
    #TODO: Replace with analytical derivatives
    np.copyto(dynamicsData.aq, -dynamicsData.pinocchio.ddq)
    np.copyto(dynamicsData.av, -dynamicsData.pinocchio.ddq)
    np.copyto(dynamicsData.gq, -dynamicsData.pinocchio.lambda_c)
    np.copyto(dynamicsData.gv, -dynamicsData.pinocchio.lambda_c)

    dynamicsData.MJtJc[:dynamicsModel.nv(),:dynamicsModel.nv()] = dynamicsData.pinocchio.M
    dynamicsData.MJtJc[:dynamicsModel.nv(),dynamicsModel.nv():] = dynamicsData.contactJ.T
    dynamicsData.MJtJc[dynamicsModel.nv():,:dynamicsModel.nv()] = dynamicsData.contactJ

    #TODO: REMOVE PINV!!!! USE DAMPED CHOLESKY
    #np.fill_diagonal(self.MJtJc, self.MJtJc.diagonal()+self.eps)
    #self.MJtJc_inv_L = np.linalg.inv(np.linalg.cholesky(dynamicsModel.MJtJc))
    #self.MJtJc_inv = np.dot(self.MJtJc_inv_L.T, self.MJtJc_inv_L)
    dynamicsData.MJtJc_inv = np.linalg.pinv(dynamicsData.MJtJc)
    dynamicsData.au = dynamicsData.MJtJc_inv[:dynamicsModel.nv(),6:dynamicsModel.nv()]
    dynamicsData.gu = dynamicsData.MJtJc_inv[dynamicsModel.nv():,6:dynamicsModel.nv()]

    # dadq #dgdq
    for i in xrange(dynamicsModel.nv()):
      dynamicsData.v_pert.fill(0.)
      dynamicsData.v_pert[i] += dynamicsData.h
      np.copyto(dynamicsData.q_pert,
                se3.integrate(dynamicsModel.pinocchio,
                              dynamicsData.x[:dynamicsModel.nq()],
                              dynamicsData.v_pert))

      dynamicsModel.computeDynamics(dynamicsData,
                                    dynamicsData.q_pert,
                                    dynamicsData.x[dynamicsModel.nq():],
                                    np.vstack([np.zeros((6,1)), dynamicsData.u]))

      dynamicsData.aq[:,i] += np.array(dynamicsData.pinocchio.ddq)[:,0]
      dynamicsData.gq[:,i] += np.array(dynamicsData.pinocchio.lambda_c)[:,0]
    dynamicsData.aq /= dynamicsData.h
    dynamicsData.gq /= dynamicsData.h

    # dadv #dgdv
    for i in xrange(dynamicsModel.nv()):
      np.copyto(dynamicsData.v_pert, dynamicsData.x[dynamicsModel.nq():])
      dynamicsData.v_pert[i] += dynamicsData.h

      dynamicsModel.computeDynamics(dynamicsData,
                                    dynamicsData.x[:dynamicsModel.nq()],
                                    dynamicsData.v_pert,
                                    np.vstack([np.zeros((6,1)), dynamicsData.u]))

      dynamicsData.av[:,i] += np.array(dynamicsData.pinocchio.ddq)[:,0]
      dynamicsData.gv[:,i] += np.array(dynamicsData.pinocchio.lambda_c)[:,0]
    dynamicsData.av /= dynamicsData.h
    dynamicsData.gv /= dynamicsData.h

  def integrateConfiguration(dynamicsModel, dynamicsData, q, dq):
    return se3.integrate(dynamicsModel.pinocchio, q, dq)

  def differenceConfiguration(dynamicsModel, dynamicsData, q0, q1):
    return se3.difference(dynamicsModel.pinocchio, q0, q1)

  def computeDynamics(dynamicsModel, dynamicsData, q, v, tau):
    # Update all terms
    #TODO: Try to reduce calculations in forward pass, and move them to backward pass
    se3.computeAllTerms(dynamicsModel.pinocchio,
                        dynamicsData.pinocchio, q, v)
    se3.updateFramePlacements(dynamicsModel.pinocchio,
                              dynamicsData.pinocchio)

    # Update the Joint jacobian and gamma
    for k, cs in enumerate(dynamicsData._contactFrameIndices):
      dynamicsData.contactJ[dynamicsModel.contactInfo.nc*k:
                            dynamicsModel.contactInfo.nc*(k+1),:] = \
        se3.getFrameJacobian(dynamicsModel.pinocchio, dynamicsData.pinocchio, cs,
                            se3.ReferenceFrame.LOCAL)[:dynamicsModel.contactInfo.nc,:]
    #TODO gamma
    dynamicsData.gamma.fill(0.)
    se3.forwardDynamics(dynamicsModel.pinocchio,
                        dynamicsData.pinocchio,
                        q, v, tau,
                        dynamicsData.contactJ, dynamicsData.gamma, 1e-8, False)