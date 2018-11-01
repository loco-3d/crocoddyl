from cddp.costs.cost import QuadraticCost
import pinocchio as se3
import numpy as np

class SE3Cost(QuadraticCost):

  def __init__(self, dynamicsModel, Mdes, weight, frame_name):
    QuadraticCost.__init__(self, dynamicsModel, Mdes, weight)
    self.dim = 6
    self.Mdes_inverse = self.ref.inverse()
    self.frame_name = frame_name
    self._frame_idx = self.dynamicsModel.pinocchioModel.getFrameId(frame_name)

    self._r = np.zeros((self.dim, 1))
    self._rx = np.zeros((self.dim, dynamicsModel.nx()))
    self._ru = np.zeros((self.dim, dynamicsModel.nu()))
    return

  def forwardRunningCalc(self, dynamicsData):
    self._r = \
        se3.log(self.Mdes_inverse*dynamicsData.pinocchioData.oMf[self._frame_idx]).vector

  def forwardTerminalCalc(self, dynamicsData):
    self.forwardRunningCalc(dynamicsData)

  def backwardRunningCalc(self, dynamicsData):
    self._rx[:,:self.dynamicsModel.nv()] = \
                                    se3.getFrameJacobian(self.dynamicsModel.pinocchioModel,
                                    dynamicsData.pinocchioData,
                                    self._frame_idx,se3.ReferenceFrame.WORLD)

  def backwardTerminalCalc(self, dynamicsData):
    self.backwardRunningCalc(dynamicsData)

  def getlu(self):
    return np.zeros((self.dynamicsModel.nu(),1))

  def getluu(self):
    return np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nu()))

  def getlux(self):
    return np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nx()))