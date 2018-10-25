from cddp.dynamics.dynamics import DynamicsModel
from cddp.dynamics.dynamics import DynamicsData
import pinocchio as se3
import numpy as np

class FloatingBaseMultibodyDynamicsData(DynamicsData):
  def __init__(self, ddpModel, t):
    DynamicsData.__init__(self, ddpModel)
    self.ddpModel = ddpModel
    self.eps = ddpModel.eps
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

    self.delta_x = np.zeros((self.dynamicsModel.nx(),1))

  def forwardRunningCalc(self):
    # Compute all terms
    #TODO: Try to reduce calculations in forward pass, and move them to backward pass
    se3.computeAllTerms(self.pinocchioModel, self.pinocchioData,
                        self.x[:self.dynamicsModel.nq()],self.x[self.dynamicsModel.nq():])
    se3.updateFramePlacements(self.pinocchioModel, self.pinocchioData)

    # Update the Joint jacobian and gamma
    for k, cs in enumerate(self._contactFrameIndices):
      self.contactJ[self.dynamicsModel.contactInfo.nc*k:
                    self.dynamicsModel.contactInfo.nc*(k+1),:]=\
        se3.getFrameJacobian(self.pinocchioModel,
                             self.pinocchioData, cs,
                             se3.ReferenceFrame.LOCAL)[:self.dynamicsModel.contactInfo.nc,:]
    #TODO gamma
    self.gamma.fill(0.)

    se3.forwardDynamics(self.dynamicsModel.pinocchioModel,
                        self.pinocchioData,self.x[:self.dynamicsModel.nq()],
                        self.x[self.dynamicsModel.nq():], np.vstack([np.zeros((6,1)),self.u]),
                        self.contactJ, self.gamma, 1e-8, False)
    return

  def forwardTerminalCalc(self):
    #TODO: Try to reduce calculations in forward pass, and move them to backward pass    
    se3.computeAllTerms(self.pinocchioModel, self.pinocchioData,
                        self.x[:self.dynamicsModel.nq()],self.x[self.dynamicsModel.nq():])
    se3.updateFramePlacements(self.pinocchioModel, self.pinocchioData)
    return

  def f(self, q, v, u):
    se3.computeAllTerms(self.pinocchioModel, self.pinocchioData,q,v)
    se3.updateFramePlacements(self.pinocchioModel, self.pinocchioData)

    # Update the Joint jacobian and gamma
    for k, cs in enumerate(self._contactFrameIndices):
      self.contactJ[self.dynamicsModel.contactInfo.nc*k:
                    self.dynamicsModel.contactInfo.nc*(k+1),:]=\
        se3.getFrameJacobian(self.pinocchioModel,
                             self.pinocchioData, cs,
                             se3.ReferenceFrame.LOCAL)[:self.dynamicsModel.contactInfo.nc,:]
    #TODO gamma
    self.gamma.fill(0.)

    se3.forwardDynamics(self.dynamicsModel.pinocchioModel,
                        self.pinocchioData,q,v, np.vstack([np.zeros((6,1)),self.u]),
                        self.contactJ, self.gamma, 1e-8, False)
    return
  
  def backwardRunningCalc(self):
    #Save the state and control in the previous iteration. Prepare for next iteration.
    np.copyto(self.x_prev,self.x)
    np.copyto(self.u_prev,self.u)
    
    #TODO: Replace with analytical derivatives
    for i in xrange(self.dynamicsModel.nv()):
      np.copyto(self.fx.aq, -self.pinocchioData.ddq)
      np.copyto(self.fx.av, -self.pinocchioData.ddq)
      np.copyto(self.gq, -self.pinocchioData.lambda_c)
      np.copyto(self.gv, -self.pinocchioData.lambda_c)

    self.MJtJc[:self.dynamicsModel.nv(),:self.dynamicsModel.nv()] = self.pinocchioData.M
    self.MJtJc[:self.dynamicsModel.nv(),self.dynamicsModel.nv():] = self.contactJ.T
    self.MJtJc[self.dynamicsModel.nv():,:self.dynamicsModel.nv()] = self.contactJ

    #np.fill_diagonal(self.MJtJc, self.MJtJc.diagonal()+self.eps)
    #self.MJtJc_inv_L = np.linalg.inv(np.linalg.cholesky(self.MJtJc))
    #self.MJtJc_inv = np.dot(self.MJtJc_inv_L.T, self.MJtJc_inv_L)

    #TODO: REMOVE PINV!!!! USE DAMPED CHOLESKY
    #print "x value", self.x.T
    self.MJtJc_inv = np.linalg.pinv(self.MJtJc)
    self.fu.au = self.MJtJc_inv[:self.dynamicsModel.nv(),6:self.dynamicsModel.nv()]
    self.gu = self.MJtJc_inv[self.dynamicsModel.nv():,6:self.dynamicsModel.nv()]

    # dadq #dgdq
    for i in xrange(self.dynamicsModel.nv()):
      self.v_pert.fill(0.)
      self.v_pert[i] += self.eps
      np.copyto(self.q_pert,
                se3.integrate(self.pinocchioModel,self.x[:self.dynamicsModel.nq()],
                              self.v_pert))
      self.f(self.q_pert,self.x[self.dynamicsModel.nq():],self.u)
      self.fx.aq[:,i] += np.array(self.pinocchioData.ddq)[:,0]
      self.gq[:,i] += np.array(self.pinocchioData.lambda_c)[:,0]
    self.fx.aq /= self.eps
    self.gq /= self.eps

    # dadv #dgdv
    for i in xrange(self.dynamicsModel.nv()):
      np.copyto(self.v_pert, self.x[self.dynamicsModel.nq():])
      self.v_pert[i] += self.eps
      self.f(self.x[:self.dynamicsModel.nq()],self.v_pert,self.u)
      self.fx.av[:,i] += np.array(self.pinocchioData.ddq)[:,0]
      self.gv[:,i] += np.array(self.pinocchioData.lambda_c)[:,0]
    self.fx.av /= self.eps
    self.gv /= self.eps
    return

  def backwardTerminalCalc(self):
    #Save the state in the previous iteration values to prepare for next iteration.
    np.copyto(self.x_prev,self.x)
    return

  def deltaX(self, x0, x1):
    self.delta_x[:self.dynamicsModel.nv()] = \
                      se3.difference(self.pinocchioModel, x0,x1)
    self.delta_x[self.dynamicsModel.nv():] = \
                              x1[self.dynamicsModel.nq():,:]- \
                              x0[self.dynamicsModel.nq():,:]
    return self.delta_x

class FloatingBaseMultibodyDynamics(DynamicsModel):

  def __init__(self, pinocchioModel, contactInfo):
    self.pinocchioModel = pinocchioModel
    self.contactInfo = contactInfo

    self._nx_impl = self.pinocchioModel.nq+self.pinocchioModel.nv
    self._nx = 2*self.pinocchioModel.nv
    self._nu = self.pinocchioModel.nv-6
    self._nq = self.pinocchioModel.nq
    self._nv = self.pinocchioModel.nv

  def createData(self, ddp_model, tInit):
    return FloatingBaseMultibodyDynamicsData(ddp_model, tInit)

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
