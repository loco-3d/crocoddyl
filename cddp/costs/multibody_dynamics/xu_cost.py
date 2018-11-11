from cddp.costs.cost import RunningQuadraticCost
import pinocchio as se3
import numpy as np


class StateCost(RunningQuadraticCost):
  """ State tracking cost.

  An important remark here is that the residual depends linearly on the state.
  So, for efficiency, we overwrite the updateQuadraticAppr function because we
  don't need 1) to find a linear approximation of the residual function, 2) to
  update the terms related to the control, and 3) to update the Hessian w.r.t
  the state (constant value).
  """
  def __init__(self, dynamicsModel, stateDes, weights):
    RunningQuadraticCost.__init__(self, dynamicsModel, stateDes, weights, dynamicsModel.nx())
    RunningQuadraticCost.__init__(self,
      dynamicsModel.nx(), dynamicsModel.nu(), dynamicsModel.nx(), weights)
    self.dynamicsModel = dynamicsModel

    np.copyto(self._rx, np.identity(self.dynamicsModel.nx()))
    np.copyto(self._data.lxx, np.diag(np.array(self.weight).squeeze()))

  def updateResidual(self, dynamicsData):
    np.copyto(self._r,
      self.dynamicsModel.deltaX(dynamicsData, self.ref, dynamicsData.x))

  def updateResidualLinearAppr(self, dynamicsData):
    # Due to the residual is equals to x, we don't need to linearize each time.
    # So, rx is defined during the construction.
    return

  def updateQuadraticAppr(self, dynamicsData):
    # We overwrite this function since this residual is equals to x. So, rx is
    # a vector of 1s, and it's not needed to multiple them.

    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(dynamicsData)

    # Updating the quadratic approximation of the cost function. We don't
    # overwrite again the lxx since is constant. This value is defined during 
    # the construction. Additionally the gradient and Hession of the cost w.r.t.
    # the control remains zero.
    np.copyto(self._data.lx, np.multiply(self.weight, self._r))


class ControlCost(RunningQuadraticCost):
  """ Control tracking cost.

  An important remark here is that the residual depends linearly on the control.
  So, for efficiency, we overwrite the updateQuadraticAppr function because we
  don't need 1) to find a linear approximation of the residual function, 2) to
  update the terms related to the state, and 3) to update the Hessian w.r.t
  the control (constant value).
  """
  def __init__(self, dynamicsModel, controlDes, weights):
    RunningQuadraticCost.__init__(self, dynamicsModel, controlDes, weights, dynamicsModel.nu())
    RunningQuadraticCost.__init__(self,
      dynamicsModel.nx(), dynamicsModel.nu(), dynamicsModel.nu(), weights)

    np.copyto(self._ru, np.identity(dynamicsModel.nu()))
    np.copyto(self._data.luu, np.diag(np.array(self.weight).squeeze()))

  def updateResidual(self, dynamicsData):
    np.copyto(self._r, dynamicsData.u-self.ref)

  def updateResidualLinearAppr(self, dynamicsData):
    # Due to the residual is equals to u, we don't need to linearize each time.
    # So, ru is defined during the construction.
    return

  def updateQuadraticAppr(self, dynamicsData):
    # We overwrite this function since this residual is equals to u. So, ru is
    # a vector of 1s, and it's not needed to multiple them.

    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(dynamicsData)

    # Updating the quadratic approximation of the cost function. We don't
    # overwrite again the luu since is constant. This value is defined during 
    # the construction. Additionally the gradient and Hession of the cost w.r.t.
    # the state remains zero.
    np.copyto(self._data.lu, np.multiply(self.weight, self._r))