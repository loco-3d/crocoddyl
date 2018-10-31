from cddp.costs.cost import QuadraticCost
import pinocchio as se3
import numpy as np

class StateCost(QuadraticCost):

  def __init__(self, dynamicsModel, stateDes, weights):
    QuadraticCost.__init__(self, dynamicsModel, stateDes, weights)
    self.pinocchioModel = dynamicsModel.pinocchioModel
    self.dim = dynamicsModel.nx()

    self._r = np.zeros((self.dim, 1))
    self._rx = np.identity(dynamicsModel.nx())
    self._ru = np.zeros((self.dim, dynamicsModel.nu()))
    self._lxx = np.diag(np.array(self.weight).squeeze())
    return

  def forwardRunningCalc(self, dynamicsData):
    self._r = self.dynamicsModel.deltaX(dynamicsData, self.ref, dynamicsData.x)

  def forwardTerminalCalc(self, dynamicsData):
    self.forwardRunningCalc(dynamicsData)

  def backwardRunningCalc(self, dynamicsData):
    return

  def backwardTerminalCalc(self, dynamicsData):
    return

  def getlu(self):
    return np.zeros((self.dynamicsModel.nu(),1))

  def getlxx(self):
    return self._lxx
  
  def getluu(self):
    return np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nu()))

  def getlux(self):
    return np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nx()))


class ControlCost(QuadraticCost):
  def __init__(self, dynamicsModel, controlDes, weights):
    QuadraticCost.__init__(self, dynamicsModel, controlDes, weights)
    self.pinocchioModel = dynamicsModel.pinocchioModel
    self.dim = dynamicsModel.nu()

    self._r = np.zeros((self.dim, 1))
    self._rx = np.zeros((self.dim, dynamicsModel.nx()))
    self._ru = np.identity(self.dim)
    self._rg = np.identity(self.dim)
    self._luu = np.diag(np.array(self.weight).squeeze())
    return

  def forwardRunningCalc(self, dynamicsData):
    np.copyto(self._r, dynamicsData.u-self.ref)

  def forwardTerminalCalc(self, dynamicsData):
    self.forwardRunningCalc(dynamicsData)

  def backwardRunningCalc(self, dynamicsData):
    return

  def backwardTerminalCalc(self, dynamicsData):
    return

  def getlx(self):
    return np.zeros((self.dynamicsModel.nx(),1))

  def getlxx(self):
    return np.zeros((self.dynamicsModel.nx(), self.dynamicsModel.nx()))

  def getluu(self):
    return self._luu

  def getlux(self):
    return np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nx()))