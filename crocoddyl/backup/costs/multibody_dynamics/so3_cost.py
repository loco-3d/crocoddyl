from crocoddyl.costs.cost import RunningQuadraticCostData
from crocoddyl.costs.cost import RunningQuadraticCost
import pinocchio as se3
import numpy as np


class SO3Task(object):
  def __init__(self, R, idx):
    self.SO3 = R
    self.idx = idx


class SO3RunningData(RunningQuadraticCostData):
  def __init__(self, dynamicModel, nr):
    # Creating the data structure for a running quadratic cost
    RunningQuadraticCostData.__init__(self, dynamicModel, nr)

    # Creating the data for the desired SE3 point and its error
    self.oRr_inv = se3.SE3(np.eye(3), np.zeros((3,1))).rotation
    self.rRf = se3.SE3(np.eye(3), np.zeros((3,1))).rotation
    self.frame_idx = 0


class SO3Cost(RunningQuadraticCost):
  """ SO3 tracking cost.

  An important remark here is that the residual only depends on the state.
  The gradient and Hessian of the cost w.r.t. the control remains zero. So, for
  efficiency, we overwrite the updateQuadraticAppr function because we don't
  need to update the terms related to the control.
  """
  def __init__(self, dynamicModel, weight):
    RunningQuadraticCost.__init__(self, 3, weight)
    self.dynamicModel = dynamicModel

  def createData(self, dynamicModel):
    return SO3RunningData(dynamicModel, self.nr)

  def updateSO3error(self, costData, dynamicData):
    """ Update the SO3 error.

    :param costData: cost data
    :param dynamicData: dynamics data
    """
    costData.rRf = \
      costData.oRr_inv * dynamicData.pinocchio.oMf[costData.frame_idx].rotation

  def updateResidual(self, costData, dynamicData, x=None, u=None):
    """ Update the residual vector.

    The SO3 error is mapped from the manifold to the algebra through the log3
    function. The residual vector represents the local angular velocity of the
    frame expressed in the reference frame, i.e. d^w^f where d is the reference
    frame and f the local frame.
    :param costData: cost data
    :param dynamicData: dynamics data
    """
    # Updating the SO3 error
    self.updateSO3error(costData, dynamicData)
    # Maping the element from the manifold to the algebra
    np.copyto(costData.r, se3.log(costData.rRf))

  def updateResidualLinearAppr(self, costData, dynamicData, x=None, u=None):
    """ Update the linear approximation of the residual.

    This correspondence to the local angular Jacobian expressed in the reference
    frame d^Ja^f = d^R^f * f^Ja^f. So we need to get the frame Jacobian in the
    local frame and them map it through d^R^f. Note that we can compute it as
    d^R^f = inv(0^R^d) * 0^R^f.
    """
    self.updateSO3error(costData, dynamicData)
    costData.rx[:,:self.dynamicModel.nv()] = \
      costData.rRf * \
      se3.getFrameJacobian(self.dynamicModel.pinocchio,
                           dynamicData.pinocchio,
                           costData.frame_idx, se3.ReferenceFrame.LOCAL)[3:6,:]

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
  def setReference(costData, frameRef):
    costData.oRr_inv = frameRef.SO3.transpose()
    costData.frame_idx = frameRef.idx