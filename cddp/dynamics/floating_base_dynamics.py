from cddp.dynamics.dynamics import DynamicsData
from cddp.dynamics.dynamics import DynamicsModel
from cddp.utils import EPS
import pinocchio as se3
import numpy as np


class FloatingBaseMultibodyDynamicsData(DynamicsData):
  def __init__(self, dynamicsModel, t, dt):
    DynamicsData.__init__(self, dynamicsModel, t, dt)

    # Pinocchio data
    self.pinocchio = dynamicsModel.pinocchio.createData()

    # Constrained dynamics data (holonomic contacts)
    nc = dynamicsModel.contactInfo.nc * dynamicsModel.contactInfo.dim(t)
    self.gq = np.zeros((nc, dynamicsModel.nv()))
    self.gv = np.zeros((nc, dynamicsModel.nv()))
    self.gu = np.zeros((nc, dynamicsModel.nu()))

    # Terms required for updatng the dynamics
    self.contactJ = np.zeros((nc, dynamicsModel.nv()))
    self.gamma = np.zeros((nc, 1))
    self._contactFrameIndices = dynamicsModel.contactInfo(t)

    # Terms required for updating the linear approximation
    self.MJtJc = np.zeros((dynamicsModel.nv() + nc,
                           dynamicsModel.nv() + nc))
    self.MJtJc_inv = np.zeros((dynamicsModel.nv() + nc,
                               dynamicsModel.nv() + nc))
    self.MJtJc_inv_L = np.zeros((dynamicsModel.nv() + nc,
                                 dynamicsModel.nv() + nc))
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

  def createData(self, t, dt):
    return FloatingBaseMultibodyDynamicsData(self, t, dt)

  def updateTerms(self, dynamicsData):
    # Compute all terms
    #TODO: Try to reduce calculations in forward pass, and move them to backward pass
    se3.computeAllTerms(self.pinocchio,
                        dynamicsData.pinocchio,
                        dynamicsData.x[:self.nq()],
                        dynamicsData.x[self.nq():])
    se3.updateFramePlacements(self.pinocchio,
                              dynamicsData.pinocchio)

  def updateDynamics(self, dynamicsData):
    # Computing the constrained forward dynamics
    self.computeDynamics(dynamicsData,
                         dynamicsData.x[:self.nq()],
                         dynamicsData.x[self.nq():],
                         np.vstack([np.zeros((6,1)), dynamicsData.u]))

    # Updating the system acceleration
    np.copyto(dynamicsData.a, dynamicsData.pinocchio.ddq)

  def updateLinearAppr(self, dynamicsData):
    #TODO: Replace with analytical derivatives
    np.copyto(dynamicsData.aq, -dynamicsData.pinocchio.ddq)
    np.copyto(dynamicsData.av, -dynamicsData.pinocchio.ddq)
    np.copyto(dynamicsData.gq, -dynamicsData.pinocchio.lambda_c)
    np.copyto(dynamicsData.gv, -dynamicsData.pinocchio.lambda_c)

    dynamicsData.MJtJc[:self.nv(),:self.nv()] = \
      dynamicsData.pinocchio.M
    dynamicsData.MJtJc[:self.nv(),self.nv():] = \
      dynamicsData.contactJ.T
    dynamicsData.MJtJc[self.nv():,:self.nv()] = \
      dynamicsData.contactJ

    #TODO: REMOVE PINV!!!! USE DAMPED CHOLESKY
    #np.fill_diagonal(self.MJtJc, self.MJtJc.diagonal()+self.eps)
    #self.MJtJc_inv_L = np.linalg.inv(np.linalg.cholesky(self.MJtJc))
    #self.MJtJc_inv = np.dot(self.MJtJc_inv_L.T, self.MJtJc_inv_L)
    np.copyto(dynamicsData.MJtJc_inv, np.linalg.pinv(dynamicsData.MJtJc))
    np.copyto(dynamicsData.au, dynamicsData.MJtJc_inv[:self.nv(),6:self.nv()])
    np.copyto(dynamicsData.gu, dynamicsData.MJtJc_inv[self.nv():,6:self.nv()])

    # dadq #dgdq
    for i in xrange(self.nv()):
      dynamicsData.v_pert.fill(0.)
      dynamicsData.v_pert[i] += dynamicsData.h
      np.copyto(dynamicsData.q_pert,
                se3.integrate(self.pinocchio,
                              dynamicsData.x[:self.nq()],
                              dynamicsData.v_pert))

      self.computeDynamics(dynamicsData,
                           dynamicsData.q_pert,
                           dynamicsData.x[self.nq():],
                           np.vstack([np.zeros((6,1)), dynamicsData.u]))

      dynamicsData.aq[:,i] += np.array(dynamicsData.pinocchio.ddq)[:,0]
      dynamicsData.gq[:,i] += np.array(dynamicsData.pinocchio.lambda_c)[:,0]
    dynamicsData.aq /= dynamicsData.h
    dynamicsData.gq /= dynamicsData.h

    # dadv #dgdv
    for i in xrange(self.nv()):
      np.copyto(dynamicsData.v_pert, dynamicsData.x[self.nq():])
      dynamicsData.v_pert[i] += dynamicsData.h

      self.computeDynamics(dynamicsData,
                           dynamicsData.x[:self.nq()],
                           dynamicsData.v_pert,
                           np.vstack([np.zeros((6,1)), dynamicsData.u]))

      dynamicsData.av[:,i] += np.array(dynamicsData.pinocchio.ddq)[:,0]
      dynamicsData.gv[:,i] += np.array(dynamicsData.pinocchio.lambda_c)[:,0]
    dynamicsData.av /= dynamicsData.h
    dynamicsData.gv /= dynamicsData.h

  def integrateConfiguration(self, q, dq):
    return se3.integrate(self.pinocchio, q, dq)

  def differenceConfiguration(self, q0, q1):
    return se3.difference(self.pinocchio, q0, q1)

  def computeDynamics(self, dynamicsData, q, v, tau):
    # Update all terms
    #TODO: Try to reduce calculations in forward pass, and move them to backward pass
    se3.computeAllTerms(self.pinocchio,
                        dynamicsData.pinocchio, q, v)
    se3.updateFramePlacements(self.pinocchio,
                              dynamicsData.pinocchio)

    # Update the Joint jacobian and gamma
    for k, cs in enumerate(dynamicsData._contactFrameIndices):
      dynamicsData.contactJ[self.contactInfo.nc*k:
                            self.contactInfo.nc*(k+1),:] = \
        se3.getFrameJacobian(
          self.pinocchio, dynamicsData.pinocchio, cs,
          se3.ReferenceFrame.LOCAL)[:self.contactInfo.nc,:]
    #TODO gamma
    dynamicsData.gamma.fill(0.)
    se3.forwardDynamics(self.pinocchio,
                        dynamicsData.pinocchio,
                        q, v, tau,
                        dynamicsData.contactJ, dynamicsData.gamma, 1e-8, False)