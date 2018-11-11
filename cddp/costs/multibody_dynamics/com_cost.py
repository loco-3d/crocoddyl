from cddp.costs.cost import RunningQuadraticCost
import numpy as np


class CoMCost(RunningQuadraticCost):
  """ CoM tracking cost.

  An important remark here is that the residual only depends on the state.
  The gradient and Hession of the cost w.r.t. the control remains zero. So, for
  efficiency, we overwrite the updateQuadraticAppr function because we don't
  need to update the terms related to the control.
  """
  def __init__(self, dynamicsModel, comDes, weights):
    RunningQuadraticCost.__init__(self, dynamicsModel, comDes, weights, 3)
    RunningQuadraticCost.__init__(self,
      dynamicsModel.nx(), dynamicsModel.nu(), 3, weights)

  def updateResidual(self, dynamicsData):
    np.copyto(self._r, dynamicsData.pinocchioData.com[0] - self.ref)

  def updateResidualLinearAppr(self, dynamicsData):
    self._rx[:,:self.dynamicsModel.nv()] = dynamicsData.pinocchioData.Jcom

  def updateQuadraticAppr(self, dynamicsData):
    # We overwrite this function since this residual function only depends on
    # state. So, the gradient and Hession of the cost w.r.t. the control remains
    # zero.
   
    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(dynamicsData)

    # Updating the quadratic approximation of the cost function
    np.copyto(self._Q_r, np.multiply(self.weight, self._r))
    np.copyto(self._Q_rx, np.multiply(self.weight, self._rx))
    np.copyto(self._lx, np.dot(self._rx.T, W_r))
    np.copyto(self._lxx, np.dot(self._rx.T, W_rx))