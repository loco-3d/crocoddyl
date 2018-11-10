from cddp.costs.cost import QuadraticCost
import numpy as np


class CoMCost(QuadraticCost):
  def __init__(self, dynamicsModel, comDes, weights):
    QuadraticCost.__init__(self, dynamicsModel, comDes, weights, 3)

  def updateResidual(self, dynamicsData):
    np.copyto(self._r, dynamicsData.pinocchioData.com[0] - self.ref)

  def updateLineaResidualModel(self, dynamicsData):
    self._rx[:,:self.dynamicsModel.nv()] = dynamicsData.pinocchioData.Jcom