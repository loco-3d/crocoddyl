from cddp.costs.cost_base import QuadraticCostBase
import numpy as np

class CoMCost(QuadraticCostBase):

  def __init__(self, dynamicsModel, comDes, weights):
    QuadraticCostBase.__init__(self, dynamicsModel, comDes, weights)
    self.dim = 3

    self._r = np.matrix(np.zeros((self.dim, 1)))
    self._rx = np.matrix(np.zeros((self.dim, dynamicsModel.nx())))
    self._ru = np.matrix(np.zeros((self.dim, dynamicsModel.nu())))
    return

  def forwardRunningCalc(self,dynamicsData):
    self._r = dynamicsData.pinocchioData.com[0] - self.ref

  def forwardTerminalCalc(self,dynamicsData):
    self.forwardRunningCalc(dynamicsData)

  def backwardRunningCalc(self, dynamicsData):
    self._rx[:,:self.dynamicsModel.nv()] = dynamicsData.pinocchioData.Jcom

  def backwardTerminalCalc(self, dynamicsData):
    self.backwardRunningCalc(dynamicsData)

  def getlux(self):
    return np.matrix(
      np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nx())))

  def getluu(self):
    return np.matrix(
      np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nu())))

  def getlgg(self):
    return np.matrix(
      np.zeros((self.dynamicsModel.dimConstraint, self.dynamicsModel.dimConstraint)))

  def getlg(self):
    return np.matrix(np.zeros((self.dynamicsModel.dimConstraint, 1)))

  def getlu(self):
    return np.matrix(np.zeros((self.dynamicsModel.nu(), 1)))
