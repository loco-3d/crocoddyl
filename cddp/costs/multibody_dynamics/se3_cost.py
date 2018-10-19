from cddp.costs.cost_base import QuadraticCostBase
import pinocchio as se3
import numpy as np

class SE3Cost(QuadraticCostBase):

  def __init__(self, dynamicsModel, Mdes, weight, frame_name):
    QuadraticCostBase.__init__(self, dynamicsModel, Mdes, weight)
    self.dim = 6
    self.Mdes_inverse = self.ref.inverse()
    self.frame_name = frame_name
    self._frame_idx = self.dynamicsModel.pinocchioModel.getFrameId(frame_name)

    self._r = np.empty((self.dim, 1))
    self._rx = np.empty((self.dim, dynamicsModel.nx()))
    self._ru = np.empty((self.dim, dynamicsModel.nu()))
    return

  def forwardRunningCalc(self,dynamicsData):
    self._r = \
        se3.log(self.Mdes_inverse*dynamicsData.pinocchioData.oMf[self._frame_idx]).vector

  def forwardTerminalCalc(self,dynamicsData):
    self.forwardRunningCalc(dynamicsData)
