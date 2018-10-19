from cddp.costs.cost_base import QuadraticCostBase
import numpy as np

class CoMCost(QuadraticCostBase):

  def __init__(self, dynamicsModel, comDes, weights):
    QuadraticCostBase.__init__(self, dynamicsModel, comDes, weights)
    self.dim = 3

    self._r = np.empty((self.dim, 1))
    self._rx = np.empty((self.dim, dynamicsModel.nx()))
    self._ru = np.empty((self.dim, dynamicsModel.nu()))
    return

  def forwardRunningCalc(self,dynamicsData):
    self._r = dynamicsData.pinocchioData.com[0] - self.ref
    
  def forwardTerminalCalc(self,dynamicsData):
    self.forwardRunningCalc(dynamicsData)
