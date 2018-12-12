from crocoddyL.costs.cost import RunningQuadraticCostData
from crocoddyL.costs.cost import RunningQuadraticCost
import numpy as np


class CoMRunningData(RunningQuadraticCostData):
  def __init__(self, dynamicModel, nr):
    # Creating the data structure for a running quadratic cost
    RunningQuadraticCostData.__init__(self, dynamicModel, nr)

    # Creating the data for the desired CoM position
    self.com_des = np.zeros((nr,1))


class CoMCost(RunningQuadraticCost):
  """ CoM tracking cost.

  An important remark here is that the residual only depends on the state.
  The gradient and Hession of the cost w.r.t. the control remains zero. So, for
  efficiency, we overwrite the updateQuadraticAppr function because we don't
  need to update the terms related to the control.
  """
  def __init__(self, dynamicModel, weights):
    RunningQuadraticCost.__init__(self, 3, weights)
    self.dynamicModel = dynamicModel

  def createData(self, dynamicModel):
    return CoMRunningData(dynamicModel, self.nr)

  def updateResidual(self, costData, dynamicData, x, u):
    np.copyto(costData.r, dynamicData.pinocchio.com[0] - costData.com_des)

  def updateResidualLinearAppr(self, costData, dynamicData, x, u):
    costData.rx[:,:self.dynamicModel.nv()] = dynamicData.pinocchio.Jcom

  def updateQuadraticAppr(self, costData, dynamicData, x, u):
    # We overwrite this function since this residual function only depends on
    # state. So, the gradient and Hession of the cost w.r.t. the control remains
    # zero.

    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(costData, dynamicData, x, u)

    # Updating the quadratic approximation of the cost function
    np.copyto(costData.Q_r, np.multiply(self.weight, costData.r))
    np.copyto(costData.Q_rx, np.multiply(self.weight, costData.rx))
    np.copyto(costData.lx, np.dot(costData.rx.T, costData.Q_r))
    np.copyto(costData.lxx, np.dot(costData.rx.T, costData.Q_rx))

  @staticmethod
  def setReference(costData, com_des):
    np.copyto(costData.com_des, com_des)