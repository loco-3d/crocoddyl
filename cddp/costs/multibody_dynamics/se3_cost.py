from cddp.costs.cost import RunningQuadraticCost
import pinocchio as se3
import numpy as np


class SE3Cost(RunningQuadraticCost):
  """ CoM tracking cost.

  An important remark here is that the residual only depends on the state.
  The gradient and Hession of the cost w.r.t. the control remains zero. So, for
  efficiency, we overwrite the updateQuadraticAppr function because we don't
  need to update the terms related to the control.
  """
  def __init__(self, dynamicsModel, Mdes, weight, frame_name):
    RunningQuadraticCost.__init__(self,
      dynamicsModel.nx(), dynamicsModel.nu(), 6, weight)
    self.dynamicsModel = dynamicsModel
    self.Mdes_inv = Mdes.inverse()
    self._frame_idx = self.dynamicsModel.pinocchioModel.getFrameId(frame_name)

  def updateResidual(self, dynamicsData):
    np.copyto(self._r,
        se3.log(self.Mdes_inv * dynamicsData.pinocchioData.oMf[self._frame_idx]).vector)

  def updateResidualLinearAppr(self, dynamicsData):
    self._rx[:,:self.dynamicsModel.nv()] = \
        se3.getFrameJacobian(self.dynamicsModel.pinocchioModel,
                             dynamicsData.pinocchioData,
                             self._frame_idx, se3.ReferenceFrame.WORLD)

  def updateQuadraticAppr(self, dynamicsData):
    # We overwrite this function since this residual function only depends on
    # state. So, the gradient and Hession of the cost w.r.t. the control remains
    # zero.

    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(dynamicsData)

    # Updating the quadratic approximation of the cost function
    np.copyto(self._Q_r, np.multiply(self.weight, self._r))
    np.copyto(self._Q_rx, np.multiply(self.weight, self._rx))
    np.copyto(self._data.lx, np.dot(self._rx.T, self._Q_r))
    np.copyto(self._data.lxx, np.dot(self._rx.T, self._Q_rx))