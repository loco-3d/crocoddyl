class DDPModel(object):
  """ Class to save the model information for the system, cost and dynamics
  """
  def __init__(self, dynamicsModel, integrator, approximator, costManager):
    self.dynamicsModel = dynamicsModel
    self.integrator = integrator
    self.approximator = approximator
    self.costManager = costManager

  def createDynamicsData(self):
    return self.dynamicsModel.createData()

  def createCostData(self):
    return self.costModel.createData()
