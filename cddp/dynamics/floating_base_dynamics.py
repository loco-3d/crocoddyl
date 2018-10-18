from cddp.dynamics.dynamics_model_base import DynamicsModelBase
from cddp.dynamics.dynamics_data_base import DynamicsDataBase

class FloatingBaseMultibodyDynamicsData(DynamicsDataBase):
  def __init__(self, dynamicsModel):
    self.pinocchioData = dynamicsModel.pinocchioModel.createData()
    
  def computeAllTerms(self):
    self.pinocchioData.computeAllTerms(x, u)

class FloatingBaseMultibodyDynamics(DynamicsModelBase):

  def __init__(self, pinocchioModel, contactInfo):
    self.pinocchioModel = pinocchioModel
    self.contactInfo = contactInfo

    self._nx = self.pinocchioModel.nq+self.pinocchioModel.nv
    self._nu = self.pinocchioModel.nv
    
  def createIntervalData(self):
    return FloatingBaseMultibodyDynamicsData(self)#.pinocchioModel,self.contactInfo)

  def nx(self):
    return self._nx

  def nu(self):
    return self._nu
