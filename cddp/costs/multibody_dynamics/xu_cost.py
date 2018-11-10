from cddp.costs.cost import QuadraticCost
import pinocchio as se3
import numpy as np


class StateCost(QuadraticCost):
  def __init__(self, dynamicsModel, stateDes, weights):
    QuadraticCost.__init__(self, dynamicsModel, stateDes, weights, 2*dynamicsModel.nv())
    self.pinocchioModel = dynamicsModel.pinocchioModel

    np.copyto(self._lxx, np.diag(np.array(self.weight).squeeze()))

  def updateResidual(self, dynamicsData):
    np.copyto(self._r, self.dynamicsModel.deltaX(dynamicsData, self.ref, dynamicsData.x))

  def updateLineaResidualModel(self, dynamicsData):
    np.copyto(self._rx, np.identity(2*self.dynamicsModel.nv()))


class ControlCost(QuadraticCost):
  def __init__(self, dynamicsModel, controlDes, weights):
    QuadraticCost.__init__(self, dynamicsModel, controlDes, weights, dynamicsModel.nu())
    self.pinocchioModel = dynamicsModel.pinocchioModel

    np.copyto(self._luu, np.diag(np.array(self.weight).squeeze()))

  def updateResidual(self, dynamicsData):
    np.copyto(self._r, dynamicsData.u-self.ref)

  def updateLineaResidualModel(self, dynamicsData):
    np.copyto(self._ru, np.identity(self.dynamicsModel.nu()))