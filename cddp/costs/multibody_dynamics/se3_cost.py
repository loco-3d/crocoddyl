from cddp.costs.cost import QuadraticCost
import pinocchio as se3
import numpy as np


class SE3Cost(QuadraticCost):
  def __init__(self, dynamicsModel, Mdes, weight, frame_name):
    QuadraticCost.__init__(self, dynamicsModel, Mdes, weight, 6)
    self.Mdes_inverse = self.ref.inverse()
    self.frame_name = frame_name
    self._frame_idx = self.dynamicsModel.pinocchioModel.getFrameId(frame_name)

  def updateResidual(self, dynamicsData):
    np.copyto(self._r,
        se3.log(self.Mdes_inverse*dynamicsData.pinocchioData.oMf[self._frame_idx]).vector)

  def updateLineaResidualModel(self, dynamicsData):
    self._rx[:,:self.dynamicsModel.nv()] = \
        se3.getFrameJacobian(self.dynamicsModel.pinocchioModel,
                             dynamicsData.pinocchioData,
                             self._frame_idx, se3.ReferenceFrame.WORLD)