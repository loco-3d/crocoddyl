from cddp.costs.cost import QuadraticCost
import numpy as np

class CoMCost(QuadraticCost):

  def __init__(self, dynamicsModel, comDes, weights):
    QuadraticCost.__init__(self, dynamicsModel, comDes, weights)
    self.dim = 3

    self._r = np.zeros((self.dim, 1))
    self._rx = np.zeros((self.dim, dynamicsModel.nx()))
    self._ru = np.zeros((self.dim, dynamicsModel.nu()))
    return

  def forwardRunningCalc(self, dynamicsData):
    self._r = dynamicsData.pinocchioData.com[0] - self.ref
    
  def forwardTerminalCalc(self, dynamicsData):
    self.forwardRunningCalc(dynamicsData)

  def backwardRunningCalc(self, dynamicsData):
    self._rx[:,:self.dynamicsModel.nv()] = dynamicsData.pinocchioData.Jcom
    
  def backwardTerminalCalc(self, dynamicsData):
    self.backwardRunningCalc(dynamicsData)

  def getlu(self):
    return np.zeros((self.dynamicsModel.nu(),1))
  
  def getluu(self):
    return np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nu()))

  def getlux(self):
    return np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nx()))