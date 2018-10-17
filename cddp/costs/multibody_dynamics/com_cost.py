from cost_base import QuadraticCostBase

class CoMCost(QuadraticCostBase):

  def __init__(self, comDes, weights):
    QuadraticCostBase.__init__(self, comDes, weights)
    return

