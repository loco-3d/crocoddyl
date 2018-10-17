from cddp.dynamics.dynamics_model_base import DynamicsModelBase


#class FloatingBaseMultibodyDynamicsData(DynamicsDataBase):

class FloatingBaseMultibodyDynamics(DynamicsModelBase):

  def __init__(self, pinocchioModel, contactInfo):
    self.pinocchiModel = pinocchioModel
    self.contactInfo = contactInfo
    #def createData(self):
    
