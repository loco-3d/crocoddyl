from cddp.dynamics.dynamics import DynamicsModel
from cddp.dynamics.dynamics import DynamicsData
from cddp.utils import EPS
import pinocchio as se3
import numpy as np


class FloatingBaseMultibodyDynamicsData(DynamicsData):
  def __init__(self, ddpModel, t):
    DynamicsData.__init__(self, ddpModel)
    self.ddpModel = ddpModel
    self.h = np.sqrt(EPS)
    self.dynamicsModel = self.ddpModel.dynamicsModel
    self.pinocchioModel = self.dynamicsModel.pinocchioModel
    self.pinocchioData = self.dynamicsModel.pinocchioModel.createData()
    self.dimConstraint = self.dynamicsModel.contactInfo.nc*\
                         self.dynamicsModel.contactInfo.dim(t)
    self.contactJ = np.zeros((self.dimConstraint,self.dynamicsModel.nv()))
    self.gamma = np.zeros((self.dimConstraint, 1))
    self._contactFrameIndices = self.dynamicsModel.contactInfo(t)

    self.MJtJc = np.zeros((self.dynamicsModel.nv()+self.dimConstraint,
                           self.dynamicsModel.nv()+self.dimConstraint))
    self.MJtJc_inv = np.zeros((self.dynamicsModel.nv()+self.dimConstraint,
                               self.dynamicsModel.nv()+self.dimConstraint))
    self.MJtJc_inv_L = np.zeros((self.dynamicsModel.nv()+self.dimConstraint,
                                 self.dynamicsModel.nv()+self.dimConstraint))
    
    self.x = np.zeros((self.dynamicsModel.nxImpl(), 1))
    self.u = np.zeros((self.dynamicsModel.nu(), 1))

    # Saving the previous iteration state and control
    self.x_prev = np.zeros((self.dynamicsModel.nxImpl(), 1))
    self.u_prev = np.zeros((self.dynamicsModel.nu(), 1))

    self.fx = self.ddpModel.discretizer.fx(self.dynamicsModel.nv(),
                                                self.dynamicsModel.nv(),
                                                self.dynamicsModel.nx(),
                                                self.dynamicsModel.nx())
    self.fu = self.ddpModel.discretizer.fu(self.dynamicsModel.nv(),
                                           self.dynamicsModel.nu(),
                                           self.dynamicsModel.nx())
    self.gx = np.zeros((self.dimConstraint, self.dynamicsModel.nx()))
    self.gu = np.zeros((self.dimConstraint, self.dynamicsModel.nu()))

    #derivative of lambda wrt q
    self.gq = np.zeros((self.dimConstraint, self.dynamicsModel.nv()))
    #derivative of lambda wrt v
    self.gv = np.zeros((self.dimConstraint, self.dynamicsModel.nv()))

    #TODO: remove these when replacing with analytical derivatives
    self.q_pert = np.zeros((self.dynamicsModel.nq(), 1))
    self.v_pert = np.zeros((self.dynamicsModel.nv(), 1))


class FloatingBaseMultibodyDynamics(DynamicsModel):

  def __init__(self, pinocchioModel, contactInfo):
    self.pinocchioModel = pinocchioModel
    self.contactInfo = contactInfo

    self._nx_impl = self.pinocchioModel.nq+self.pinocchioModel.nv
    self._nx = 2*self.pinocchioModel.nv
    self._nu = self.pinocchioModel.nv-6
    self._nq = self.pinocchioModel.nq
    self._nv = self.pinocchioModel.nv
    self.delta_x = np.zeros((self._nx,1))

  def createData(self, ddpModel, tInit):
    return FloatingBaseMultibodyDynamicsData(ddpModel, tInit)

  def nx(self):
    return self._nx

  def nxImpl(self):
    return self._nx_impl

  def nu(self):
    return self._nu

  def nq(self):
    return self._nq

  def nv(self):
    return self._nv

  def forwardRunningCalc(self, dynamicsData):
    # Compute all terms
    #TODO: Try to reduce calculations in forward pass, and move them to backward pass
    se3.computeAllTerms(self.pinocchioModel, dynamicsData.pinocchioData,
                        dynamicsData.x[:self.nq()],
                        dynamicsData.x[self.nq():])
    se3.updateFramePlacements(self.pinocchioModel, dynamicsData.pinocchioData)

    # Update the Joint jacobian and gamma
    for k, cs in enumerate(dynamicsData._contactFrameIndices):
      dynamicsData.contactJ[self.contactInfo.nc*k:
                            self.contactInfo.nc*(k+1),:] = \
        se3.getFrameJacobian(self.pinocchioModel, dynamicsData.pinocchioData, cs,
                             se3.ReferenceFrame.LOCAL)[:self.contactInfo.nc,:]
    #TODO gamma
    dynamicsData.gamma.fill(0.)

    se3.forwardDynamics(self.pinocchioModel, dynamicsData.pinocchioData,
                        dynamicsData.x[:self.nq()], dynamicsData.x[self.nq():],
                        np.vstack([np.zeros((6,1)), dynamicsData.u]),
                        dynamicsData.contactJ, dynamicsData.gamma, 1e-8, False)
    return

  def forwardTerminalCalc(self, dynamicsData):
    #TODO: Try to reduce calculations in forward pass, and move them to backward pass    
    se3.computeAllTerms(self.pinocchioModel, dynamicsData.pinocchioData,
                        dynamicsData.x[:self.nq()], dynamicsData.x[self.nq():])
    se3.updateFramePlacements(self.pinocchioModel, dynamicsData.pinocchioData)
    return

  def f(self, dynamicsData, q, v, u):
    se3.computeAllTerms(self.pinocchioModel, dynamicsData.pinocchioData, q, v)
    se3.updateFramePlacements(self.pinocchioModel, dynamicsData.pinocchioData)

    # Update the Joint jacobian and gamma
    for k, cs in enumerate(dynamicsData._contactFrameIndices):
      dynamicsData.contactJ[self.contactInfo.nc*k:
                            self.contactInfo.nc*(k+1),:] = \
        se3.getFrameJacobian(self.pinocchioModel, dynamicsData.pinocchioData, cs,
                             se3.ReferenceFrame.LOCAL)[:self.contactInfo.nc,:]
    #TODO gamma
    dynamicsData.gamma.fill(0.)

    se3.forwardDynamics(self.pinocchioModel, dynamicsData.pinocchioData,
                        q, v, np.vstack([np.zeros((6,1)), u]),
                        dynamicsData.contactJ, dynamicsData.gamma, 1e-8, False)
    return
  
  def backwardRunningCalc(self, dynamicsData):
    #Save the state and control in the previous iteration. Prepare for next iteration.
    # TODO move to the DDP solver
    np.copyto(dynamicsData.x_prev, dynamicsData.x)
    np.copyto(dynamicsData.u_prev, dynamicsData.u)
    
    #TODO: Replace with analytical derivatives
    for i in xrange(self.nv()):
      np.copyto(dynamicsData.fx.aq, -dynamicsData.pinocchioData.ddq)
      np.copyto(dynamicsData.fx.av, -dynamicsData.pinocchioData.ddq)
      np.copyto(dynamicsData.gq, -dynamicsData.pinocchioData.lambda_c)
      np.copyto(dynamicsData.gv, -dynamicsData.pinocchioData.lambda_c)

    dynamicsData.MJtJc[:self.nv(),:self.nv()] = dynamicsData.pinocchioData.M
    dynamicsData.MJtJc[:self.nv(),self.nv():] = dynamicsData.contactJ.T
    dynamicsData.MJtJc[self.nv():,:self.nv()] = dynamicsData.contactJ

    #np.fill_diagonal(self.MJtJc, self.MJtJc.diagonal()+self.eps)
    #self.MJtJc_inv_L = np.linalg.inv(np.linalg.cholesky(self.MJtJc))
    #self.MJtJc_inv = np.dot(self.MJtJc_inv_L.T, self.MJtJc_inv_L)

    #TODO: REMOVE PINV!!!! USE DAMPED CHOLESKY
    #print "x value", self.x.T
    dynamicsData.MJtJc_inv = np.linalg.pinv(dynamicsData.MJtJc)
    dynamicsData.fu.au = dynamicsData.MJtJc_inv[:self.nv(),6:self.nv()]
    dynamicsData.gu = dynamicsData.MJtJc_inv[self.nv():,6:self.nv()]

    # dadq #dgdq
    for i in xrange(self.nv()):
      dynamicsData.v_pert.fill(0.)
      dynamicsData.v_pert[i] += dynamicsData.h
      np.copyto(dynamicsData.q_pert,
                se3.integrate(self.pinocchioModel,
                              dynamicsData.x[:self.nq()],
                              dynamicsData.v_pert))
      self.f(dynamicsData, dynamicsData.q_pert, dynamicsData.x[self.nq():], dynamicsData.u)
      dynamicsData.fx.aq[:,i] += np.array(dynamicsData.pinocchioData.ddq)[:,0]
      dynamicsData.gq[:,i] += np.array(dynamicsData.pinocchioData.lambda_c)[:,0]
    dynamicsData.fx.aq /= dynamicsData.h
    dynamicsData.gq /= dynamicsData.h

    # dadv #dgdv
    for i in xrange(self.nv()):
      np.copyto(dynamicsData.v_pert, dynamicsData.x[self.nq():])
      dynamicsData.v_pert[i] += dynamicsData.h
      self.f(dynamicsData, dynamicsData.x[:self.nq()], dynamicsData.v_pert, dynamicsData.u)
      dynamicsData.fx.av[:,i] += np.array(dynamicsData.pinocchioData.ddq)[:,0]
      dynamicsData.gv[:,i] += np.array(dynamicsData.pinocchioData.lambda_c)[:,0]
    dynamicsData.fx.av /= dynamicsData.h
    dynamicsData.gv /= dynamicsData.h
    return

  def backwardTerminalCalc(self, dynamicsData):
    #Save the state in the previous iteration values to prepare for next iteration.
    # TODO move to the solver
    np.copyto(dynamicsData.x_prev, dynamicsData.x)
    return

  def deltaX(self, x0, x1):
    self.delta_x[:self.nv()] = \
                      se3.difference(self.pinocchioModel,
                                     x0[:self.nq()], x1[:self.nq()])
    self.delta_x[self.nv():] = \
                              x1[self.nq():,:]- \
                              x0[self.nq():,:]
    return self.delta_x