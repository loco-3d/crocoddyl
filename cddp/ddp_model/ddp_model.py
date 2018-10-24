class DDPModel(object):
  """ Class to save the model information for the system, cost and dynamics
  """
  def __init__(self, dynamicsModel, integrator, discretizer, costManager):
    self.dynamicsModel = dynamicsModel
    self.integrator = integrator
    self.discretizer = discretizer
    self.costManager = costManager
    #TODO: Move to proper location
    self.eps = -1.

  def createRunningDynamicsData(self, tInit):
    return self.dynamicsModel.createData(self, tInit)

  def createRunningCostData(self):
    return self.costManager.createRunningData(self)

  def createTerminalDynamicsData(self, tFinal):
    return self.dynamicsModel.createData(self, tFinal)

  def createTerminalCostData(self):
    return self.costManager.createTerminalData(self)
