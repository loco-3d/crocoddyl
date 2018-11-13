from cddp.costs.cost import RunningQuadraticCostData
from cddp.costs.cost import RunningQuadraticCost
import pinocchio as se3
import numpy as np


class SE3RunningData(RunningQuadraticCostData):
  def __init__(self, nx, nu, nr):
    # Creating the data structure for a running quadratic cost
    RunningQuadraticCostData.__init__(self, nx, nu, nr)

    # Creating the data for the desired SE3 point
    self.Mdes_inv = se3.SE3()
    self.frame_idx = 0


class SE3Cost(RunningQuadraticCost):
  """ CoM tracking cost.

  An important remark here is that the residual only depends on the state.
  The gradient and Hession of the cost w.r.t. the control remains zero. So, for
  efficiency, we overwrite the updateQuadraticAppr function because we don't
  need to update the terms related to the control.
  """
  def __init__(self, dynamicsModel, weight, Mdes, frame_name):
    RunningQuadraticCost.__init__(self, 6, weight)
    self.dynamicsModel = dynamicsModel
    self.Mdes_inv = Mdes.inverse()
    self.frame_idx = self.dynamicsModel.pinocchioModel.getFrameId(frame_name)

  def createData(self, nx, nu = 0):
    # A default value of nu allows us to use this class as terminal one
    # return SE3RunningData(nx, nu)
    data = SE3RunningData(nx, nu, self.nr)
    #TODO set the externally the desired state
    data.Mdes_inv = self.Mdes_inv
    data.frame_idx =self.frame_idx
    return data

  def updateResidual(self, costData, dynamicsData):
    np.copyto(costData.r,
        se3.log(costData.Mdes_inv * dynamicsData.pinocchioData.oMf[costData.frame_idx]).vector)

  def updateResidualLinearAppr(self, costData, dynamicsData):
    costData.rx[:,:self.dynamicsModel.nv()] = \
        se3.getFrameJacobian(self.dynamicsModel.pinocchioModel,
                             dynamicsData.pinocchioData,
                             costData.frame_idx, se3.ReferenceFrame.WORLD)

  def updateQuadraticAppr(self, costData, dynamicsData):
    # We overwrite this function since this residual function only depends on
    # state. So, the gradient and Hession of the cost w.r.t. the control remains
    # zero.

    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(costData, dynamicsData)

    # Updating the quadratic approximation of the cost function
    np.copyto(costData.Q_r, np.multiply(self.weight, costData.r))
    np.copyto(costData.Q_rx, np.multiply(self.weight, costData.rx))
    np.copyto(costData.lx, np.dot(costData.rx.T, costData.Q_r))
    np.copyto(costData.lxx, np.dot(costData.rx.T, costData.Q_rx))
