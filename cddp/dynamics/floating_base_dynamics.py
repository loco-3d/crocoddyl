from cddp.dynamics.dynamics_model_base import DynamicsModelBase
from cddp.dynamics.dynamics_data_base import DynamicsDataBase
import pinocchio as se3
import numpy as np

class FloatingBaseMultibodyDynamicsData(DynamicsDataBase):
  def __init__(self, dynamicsModel, t):
    DynamicsDataBase.__init__(self, dynamicsModel)
    self.dynamicsModel = dynamicsModel
    self.pinocchioModel = dynamicsModel.pinocchioModel
    self.pinocchioData = dynamicsModel.pinocchioModel.createData()
    self.dimConstraint = dynamicsModel.contactInfo.nc*dynamicsModel.contactInfo.dim(t)
    self.contactJ = np.empty((self.dimConstraint,dynamicsModel.nv()))
    self.gamma = np.empty((self.dimConstraint, 1))

    self.x = np.empty((dynamicsModel.nxImpl(), 1))
    self.u = np.empty((dynamicsModel.nu(), 1))
    
    self._contactFrameIndices = dynamicsModel.contactInfo(t)

  def computeAllTerms(self):
    self.pinocchioData.computeAllTerms(x, u)

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
  
class FloatingBaseMultibodyDynamics(DynamicsModelBase):

  def __init__(self, pinocchioModel, contactInfo):
    self.pinocchioModel = pinocchioModel
    self.contactInfo = contactInfo

    self._nx_impl = self.pinocchioModel.nq+self.pinocchioModel.nv
    self._nx = 2*self.pinocchioModel.nv
    self._nu = self.pinocchioModel.nv-6
    self._nq = self.pinocchioModel.nq
    self._nv = self.pinocchioModel.nv

  def createIntervalData(self, tInit):
    return FloatingBaseMultibodyDynamicsData(self, tInit)

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
