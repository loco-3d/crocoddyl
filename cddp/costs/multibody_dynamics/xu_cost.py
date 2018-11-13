from cddp.costs.cost import RunningQuadraticCostData
from cddp.costs.cost import RunningQuadraticCost
import pinocchio as se3
import numpy as np


class StateRunningData(RunningQuadraticCostData):
  def __init__(self, nx, nu, nr):
    # Creating the data structure for a running quadratic cost
    RunningQuadraticCostData.__init__(self, nx, nu, nr)

    # Creating the data for the desired state
    self.x_des = np.zeros((37,1))


class ControlRunningData(RunningQuadraticCostData):
  def __init__(self, nx, nu, nr):
    # Creating the data structure for a running quadratic cost
    RunningQuadraticCostData.__init__(self, nx, nu, nr)

    # Creating the data for the desired control
    self.u_des = np.zeros((nr,1))


class StateCost(RunningQuadraticCost):
  """ State tracking cost.

  An important remark here is that the residual depends linearly on the state.
  So, for efficiency, we overwrite the updateQuadraticAppr function because we
  don't need 1) to find a linear approximation of the residual function, 2) to
  update the terms related to the control, and 3) to update the Hessian w.r.t
  the state (constant value).
  """
  def __init__(self, dynamicsModel, weights, x_des):
    RunningQuadraticCost.__init__(self, dynamicsModel.nx(), weights)
    self.dynamicsModel = dynamicsModel
    self.x_des = x_des

  def createData(self, nx, nu):
    data = StateRunningData(nx, nu, self.nr)
    np.copyto(data.rx, np.identity(nx))
    np.copyto(data.lxx, np.diag(np.array(self.weight).squeeze()))
    np.copyto(data.x_des, self.x_des) #TODO set the externally the desired state
    return data

  def updateResidual(self, costData, dynamicsData):
    np.copyto(costData.r,
      self.dynamicsModel.deltaX(dynamicsData, costData.x_des, dynamicsData.x))

  def updateResidualLinearAppr(self, costData, dynamicsData):
    # Due to the residual is equals to x, we don't need to linearize each time.
    # So, rx is defined during the construction.
    return

  def updateQuadraticAppr(self, costData, dynamicsData):
    # We overwrite this function since this residual is equals to x. So, rx is
    # a vector of 1s, and it's not needed to multiple them.

    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(costData, dynamicsData)

    # Updating the quadratic approximation of the cost function. We don't
    # overwrite again the lxx since is constant. This value is defined during
    # the construction. Additionally the gradient and Hession of the cost w.r.t.
    # the control remains zero.
    np.copyto(costData.lx, np.multiply(self.weight, costData.r))



class ControlCost(RunningQuadraticCost):
  """ Control tracking cost.

  An important remark here is that the residual depends linearly on the control.
  So, for efficiency, we overwrite the updateQuadraticAppr function because we
  don't need 1) to find a linear approximation of the residual function, 2) to
  update the terms related to the state, and 3) to update the Hessian w.r.t
  the control (constant value).
  """
  def __init__(self, dynamicsModel, weights, u_des):
    RunningQuadraticCost.__init__(self, dynamicsModel.nu(), weights)
    self.u_des = u_des

  def createData(self, nx, nu):
    data = ControlRunningData(nx, nu, self.nr)
    np.copyto(data.ru, np.identity(nu))
    np.copyto(data.luu, np.diag(np.array(self.weight).squeeze()))
    np.copyto(data.u_des, self.u_des) #TODO set the externally the desired state
    return data

  def updateResidual(self, costData, dynamicsData):
    np.copyto(costData.r, dynamicsData.u - costData.u_des)

  def updateResidualLinearAppr(self, costData, dynamicsData):
    # Due to the residual is equals to u, we don't need to linearize each time.
    # So, ru is defined during the construction.
    return

  def updateQuadraticAppr(self, costData, dynamicsData):
    # We overwrite this function since this residual is equals to u. So, ru is
    # a vector of 1s, and it's not needed to multiple them.

    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(costData, dynamicsData)

    # Updating the quadratic approximation of the cost function. We don't
    # overwrite again the luu since is constant. This value is defined during
    # the construction. Additionally the gradient and Hession of the cost w.r.t.
    # the state remains zero.
    np.copyto(costData.lu, np.multiply(self.weight, costData.r))
