from cddp.dynamics.dynamics import DynamicsModel
from cddp.dynamics.dynamics import DynamicsData
from cddp.utils import EPS
import pinocchio as se3
import numpy as np


class FloatingBaseMultibodyDynamicsData(DynamicsData):
  def __init__(self, dynamicsModel, t, dt):
    DynamicsData.__init__(self, dynamicsModel, dt)
    self.h = np.sqrt(EPS)
    self.pinocchio = dynamicsModel.pinocchio.createData()
    self.dimConstraint = dynamicsModel.contactInfo.nc*\
                         dynamicsModel.contactInfo.dim(t)
    self.contactJ = np.zeros((self.dimConstraint, dynamicsModel.nv()))
    self.gamma = np.zeros((self.dimConstraint, 1))
    self._contactFrameIndices = dynamicsModel.contactInfo(t)

    self.MJtJc = np.zeros((dynamicsModel.nv() + self.dimConstraint,
                           dynamicsModel.nv() + self.dimConstraint))
    self.MJtJc_inv = np.zeros((dynamicsModel.nv() + self.dimConstraint,
                               dynamicsModel.nv() + self.dimConstraint))
    self.MJtJc_inv_L = np.zeros((dynamicsModel.nv() + self.dimConstraint,
                                 dynamicsModel.nv() + self.dimConstraint))

    self.gq = np.zeros((self.dimConstraint, dynamicsModel.nv()))
    self.gv = np.zeros((self.dimConstraint, dynamicsModel.nv()))
    self.gu = np.zeros((self.dimConstraint, dynamicsModel.nu()))

    #TODO: remove these when replacing with analytical derivatives
    self.q_pert = np.zeros((dynamicsModel.nq(), 1))
    self.v_pert = np.zeros((dynamicsModel.nv(), 1))



class FloatingBaseMultibodyDynamics(DynamicsModel):
  def __init__(self, pinocchioModel, discretizer, contactInfo):
    DynamicsModel.__init__(self, pinocchioModel.nq + pinocchioModel.nv,
                           2 * pinocchioModel.nv,
                           pinocchioModel.nv - 6)
    self.pinocchio = pinocchioModel
    self.discretizer = discretizer
    self.contactInfo = contactInfo

    self._nq = self.pinocchio.nq
    self._nv = self.pinocchio.nv

  def createData(dynamicsModel, tInit, dt):
    return FloatingBaseMultibodyDynamicsData(dynamicsModel, tInit, dt)

  def updateDynamics(dynamicsModel, dynamicsData, q, v):
    # Compute all terms
    #TODO: Try to reduce calculations in forward pass, and move them to backward pass
    se3.computeAllTerms(dynamicsModel.pinocchio,
                        dynamicsData.pinocchio, q, v)
    se3.updateFramePlacements(dynamicsModel.pinocchio,
                              dynamicsData.pinocchio)

  def computeDynamics(dynamicsModel, dynamicsData, q, v, tau):
    # Update the dynamics
    dynamicsModel.updateDynamics(dynamicsData, q, v)

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

  def forwardRunningCalc(dynamicsModel, dynamicsData):
    dynamicsModel.computeDynamics(dynamicsData,
                                  dynamicsData.x[:dynamicsModel.nq()],
                                  dynamicsData.x[dynamicsModel.nq():],
                                  np.vstack([np.zeros((6,1)), dynamicsData.u]))
    return

  def forwardTerminalCalc(dynamicsModel, dynamicsData):
    dynamicsModel.updateDynamics(dynamicsData,
                                 dynamicsData.x[:dynamicsModel.nq()],
                                 dynamicsData.x[dynamicsModel.nq():])
    return

  def backwardRunningCalc(dynamicsModel, dynamicsData):
    #TODO: Replace with analytical derivatives
    np.copyto(dynamicsData.aq, -dynamicsData.pinocchio.ddq)
    np.copyto(dynamicsData.av, -dynamicsData.pinocchio.ddq)
    np.copyto(dynamicsData.gq, -dynamicsData.pinocchio.lambda_c)
    np.copyto(dynamicsData.gv, -dynamicsData.pinocchio.lambda_c)

    dynamicsData.MJtJc[:dynamicsModel.nv(),:dynamicsModel.nv()] = dynamicsData.pinocchio.M
    dynamicsData.MJtJc[:dynamicsModel.nv(),dynamicsModel.nv():] = dynamicsData.contactJ.T
    dynamicsData.MJtJc[dynamicsModel.nv():,:dynamicsModel.nv()] = dynamicsData.contactJ

    #np.fill_diagonal(self.MJtJc, self.MJtJc.diagonal()+self.eps)
    #self.MJtJc_inv_L = np.linalg.inv(np.linalg.cholesky(dynamicsModel.MJtJc))
    #self.MJtJc_inv = np.dot(self.MJtJc_inv_L.T, self.MJtJc_inv_L)

    #TODO: REMOVE PINV!!!! USE DAMPED CHOLESKY
    #print "x value", self.x.T
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

    # Discretizing the dynamics
    dynamicsModel.discretizer.backwardRunningCalc(dynamicsModel, dynamicsData)
    return

  def backwardTerminalCalc(dynamicsModel, dynamicsData):
    return

  def nq(self):
    return self._nq

  def nv(self):
    return self._nv

  def deltaX(dynamicsModel, dynamicsData, x0, x1):
    dynamicsData.diff_x[:dynamicsModel.nv()] = \
        se3.difference(dynamicsModel.pinocchio,
                       x0[:dynamicsModel.nq()], x1[:dynamicsModel.nq()])
    dynamicsData.diff_x[dynamicsModel.nv():] = \
        x1[dynamicsModel.nq():,:] - x0[dynamicsModel.nq():,:]
    return dynamicsData.diff_x