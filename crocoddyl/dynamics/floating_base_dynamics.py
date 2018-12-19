from crocoddyl.dynamics.dynamics import DynamicData
from crocoddyl.dynamics.dynamics import DynamicModel
from crocoddyl.utils import EPS
import pinocchio as se3
import numpy as np


class FloatingBaseMultibodyDynamicData(DynamicData):
  def __init__(self, dynamicModel, t, dt):
    DynamicData.__init__(self, dynamicModel, t, dt)

    # Pinocchio data
    self.pinocchio = dynamicModel.pinocchio.createData()

    # Constrained dynamics data (holonomic contacts)
    nc = dynamicModel.contactInfo.nc * dynamicModel.contactInfo.dim(t)
    self.gq = np.zeros((nc, dynamicModel.nv()))
    self.gv = np.zeros((nc, dynamicModel.nv()))
    self.gu = np.zeros((nc, dynamicModel.nu()))

    # Terms required for updating the dynamics
    self.Jc = np.zeros((nc, dynamicModel.nv()))
    self.a_ref = np.zeros((nc, 1))
    self.contactFrames = dynamicModel.contactInfo(t)

    # Terms required for updating the linear approximation
    self.MJtJc = np.zeros((dynamicModel.nv() + nc,
                           dynamicModel.nv() + nc))
    self.MJtJc_inv = np.zeros((dynamicModel.nv() + nc,
                               dynamicModel.nv() + nc))
    self.MJtJc_inv_L = np.zeros((dynamicModel.nv() + nc,
                                 dynamicModel.nv() + nc))
    #TODO: remove these when replacing with analytical derivatives
    self.h = np.sqrt(EPS)
    self.q_pert = np.zeros((dynamicModel.nq(), 1))
    self.v_pert = np.zeros((dynamicModel.nv(), 1))


class FloatingBaseMultibodyDynamics(DynamicModel):
  def __init__(self, integrator, discretizer, pinocchioModel, contactInfo):
    DynamicModel.__init__(self, integrator, discretizer,
                           pinocchioModel.nq,
                           pinocchioModel.nv,
                           pinocchioModel.nv - 6)
    self.pinocchio = pinocchioModel
    self.contactInfo = contactInfo

  def createData(self, t, dt):
    return FloatingBaseMultibodyDynamicData(self, t, dt)

  def updateTerms(self, dynamicData, x):
    # Compute all terms
    #TODO: Try to reduce calculations in forward pass, and move them to backward pass
    se3.computeAllTerms(self.pinocchio, dynamicData.pinocchio,
                        x[:self.nq()], x[self.nq():])
    se3.updateFramePlacements(self.pinocchio,
                              dynamicData.pinocchio)

  def updateDynamics(self, dynamicData, x, u):
    # Computing the constrained forward dynamics
    self.computeDynamics(dynamicData,
                         x[:self.nq()], x[self.nq():],
                         np.vstack([np.zeros((6,1)), u]))

    # Updating the system acceleration
    np.copyto(dynamicData.a, dynamicData.pinocchio.ddq)

  def updateLinearAppr(self, dynamicData, x, u):
    #TODO: Replace with analytical derivatives
    np.copyto(dynamicData.aq, -dynamicData.pinocchio.ddq)
    np.copyto(dynamicData.av, -dynamicData.pinocchio.ddq)
    np.copyto(dynamicData.gq, -dynamicData.pinocchio.lambda_c)
    np.copyto(dynamicData.gv, -dynamicData.pinocchio.lambda_c)

    dynamicData.MJtJc[:self.nv(),:self.nv()] = \
      dynamicData.pinocchio.M
    dynamicData.MJtJc[:self.nv(),self.nv():] = \
      dynamicData.Jc.T
    dynamicData.MJtJc[self.nv():,:self.nv()] = \
      dynamicData.Jc

    #TODO: REMOVE PINV!!!! USE DAMPED CHOLESKY
    #np.fill_diagonal(self.MJtJc, self.MJtJc.diagonal()+self.eps)
    #self.MJtJc_inv_L = np.linalg.inv(np.linalg.cholesky(self.MJtJc))
    #self.MJtJc_inv = np.dot(self.MJtJc_inv_L.T, self.MJtJc_inv_L)
    np.copyto(dynamicData.MJtJc_inv, np.linalg.pinv(dynamicData.MJtJc))
    np.copyto(dynamicData.au, dynamicData.MJtJc_inv[:self.nv(),6:self.nv()])
    np.copyto(dynamicData.gu, dynamicData.MJtJc_inv[self.nv():,6:self.nv()])

    # dadq #dgdq
    for i in xrange(self.nv()):
      dynamicData.v_pert.fill(0.)
      dynamicData.v_pert[i] += dynamicData.h
      np.copyto(dynamicData.q_pert,
        se3.integrate(self.pinocchio, x[:self.nq()], dynamicData.v_pert))

      self.computeDynamics(dynamicData,
                           dynamicData.q_pert,
                           x[self.nq():],
                           np.vstack([np.zeros((6,1)), u]))

      dynamicData.aq[:,i] += np.array(dynamicData.pinocchio.ddq)[:,0]
      dynamicData.gq[:,i] += np.array(dynamicData.pinocchio.lambda_c)[:,0]
    dynamicData.aq /= dynamicData.h
    dynamicData.gq /= dynamicData.h

    # dadv #dgdv
    for i in xrange(self.nv()):
      np.copyto(dynamicData.v_pert, x[self.nq():])
      dynamicData.v_pert[i] += dynamicData.h

      self.computeDynamics(dynamicData,
                           x[:self.nq()],
                           dynamicData.v_pert,
                           np.vstack([np.zeros((6,1)), u]))

      dynamicData.av[:,i] += np.array(dynamicData.pinocchio.ddq)[:,0]
      dynamicData.gv[:,i] += np.array(dynamicData.pinocchio.lambda_c)[:,0]
    dynamicData.av /= dynamicData.h
    dynamicData.gv /= dynamicData.h

  def integrateConfiguration(self, q, dq):
    return se3.integrate(self.pinocchio, q, dq)

  def differenceConfiguration(self, q0, q1):
    return se3.difference(self.pinocchio, q0, q1)

  def computeDynamics(self, dynamicData, q, v, tau):
    # Update all terms
    #TODO: Try to reduce calculations in forward pass, and move them to backward pass
    se3.computeAllTerms(self.pinocchio,
                        dynamicData.pinocchio, q, v)
    se3.updateFramePlacements(self.pinocchio,
                              dynamicData.pinocchio)

    # Update the Joint Jacobian and the reference acceleration
    for k, frame_id in enumerate(dynamicData.contactFrames):
      # Computing the frame Jacobian in the local frame
      dynamicData.Jc[self.contactInfo.nc*k:
                           self.contactInfo.nc*(k+1),:] = \
        se3.getFrameJacobian(
          self.pinocchio, dynamicData.pinocchio, frame_id,
          se3.ReferenceFrame.LOCAL)[:self.contactInfo.nc,:]

      # Mapping the reference acceleration into the local frame
      # TODO define a good interface for contact phase and then remove this hard
      # coded if-rule
      if self.contactInfo.nc == 3:
        dynamicData.a_ref[self.contactInfo.nc*k:
                          self.contactInfo.nc*(k+1)] = \
          se3.getFrameAcceleration(self.pinocchio, dynamicData.pinocchio, frame_id).linear
      elif nc == 6:
        dynamicData.a_ref[self.contactInfo.nc*k:
                          self.contactInfo.nc*(k+1)] = \
          se3.getFrameAcceleration(self.pinocchio, dynamicData.pinocchio, frame_id).vector
      else:
        print "nc has to be equals to 3 or 6"

    se3.forwardDynamics(self.pinocchio,
                        dynamicData.pinocchio,
                        q, v, tau,
                        dynamicData.Jc, dynamicData.a_ref, 1e-8, False)