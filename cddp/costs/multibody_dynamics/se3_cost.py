from cddp.costs.cost import RunningQuadraticCostData
from cddp.costs.cost import RunningQuadraticCost
import pinocchio as se3
import numpy as np


class SE3Task(object):
  def __init__(self, M, idx):
    self.SE3 = M
    self.idx = idx


class SE3RunningData(RunningQuadraticCostData):
  def __init__(self, dynamicModel, nr):
    # Creating the data structure for a running quadratic cost
    RunningQuadraticCostData.__init__(self, dynamicModel, nr)

    # Creating the data for the desired SE3 point and its error
    self.oMr_inv = se3.SE3()
    self.rMf = se3.SE3()
    self.frame_idx = 0


class SE3Cost(RunningQuadraticCost):
  """ CoM tracking cost.

  An important remark here is that the residual only depends on the state.
  The gradient and Hessian of the cost w.r.t. the control remains zero. So, for
  efficiency, we overwrite the updateQuadraticAppr function because we don't
  need to update the terms related to the control.
  """
  def __init__(self, dynamicModel, weight):
    RunningQuadraticCost.__init__(self, 6, weight)
    self.dynamicModel = dynamicModel

  def createData(self, dynamicModel):
    return SE3RunningData(dynamicModel, self.nr)

  def updateSE3error(self, costData, dynamicsData):
    """ Update the SE3 error.

    :param costData: cost data
    :param dynamicsData: dynamics data
    """
    costData.rMf = costData.oMr_inv * dynamicsData.pinocchio.oMf[costData.frame_idx]

  def updateResidual(self, costData, dynamicsData, x, u):
    """ Update the residual vector.

    The SE3 error is mapped from the manifold to the algebra through the log
    function. The residual vector represents the local velocity of the frame
    expressed in the reference frame, i.e. d^V^f where d is the reference frame
    and f the local frame.
    :param costData: cost data
    :param dynamicsData: dynamics data
    """
    # Updating the SE3 error
    self.updateSE3error(costData, dynamicsData)
    # Maping the element from the manifold to the algebra
    np.copyto(costData.r,
        se3.log(costData.rMf).vector)

  def updateResidualLinearAppr(self, costData, dynamicsData, x, u):
    """ Update the linear approximation of the residual.

    This correspondence to the local Jacobian expressed in the reference frame
    d^J^f = d^X^f * f^J^f. So we need to get the frame Jacobian in the local
    frame and them map it through d^X^f. Note that we can compute it as 
    d^X^f = inv(0^X^d) * 0^X^f.
    """
    self.updateSE3error(costData, dynamicsData)
    costData.rx[:,:self.dynamicModel.nv()] = \
        costData.rMf.action * \
        se3.getFrameJacobian(self.dynamicModel.pinocchio,
                             dynamicsData.pinocchio,
                             costData.frame_idx, se3.ReferenceFrame.LOCAL)

  def updateQuadraticAppr(self, costData, dynamicsData, x, u):
    # We overwrite this function since this residual function only depends on
    # state. So, the gradient and Hession of the cost w.r.t. the control remains
    # zero.

    # Updating the linear approximation of the residual function
    self.updateResidualLinearAppr(costData, dynamicsData, x, u)

    # Updating the quadratic approximation of the cost function
    np.copyto(costData.Q_r, np.multiply(self.weight, costData.r))
    np.copyto(costData.Q_rx, np.multiply(self.weight, costData.rx))
    np.copyto(costData.lx, np.dot(costData.rx.T, costData.Q_r))
    np.copyto(costData.lxx, np.dot(costData.rx.T, costData.Q_rx))

  @staticmethod
  def setReference(costData, frameRef):
    costData.oMr_inv = frameRef.SE3.inverse()
    costData.frame_idx = frameRef.idx