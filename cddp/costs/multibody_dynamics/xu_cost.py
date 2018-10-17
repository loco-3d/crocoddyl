from cost_base import CostBase

class StateCost(CostBase):

  def __init__(self, stateDes, weights):
    CostBase.__init__(self, stateDes, weights)
    return

class ControlCost(CostBase):

  def __init__(self, controlDes, weights):
    CostBase.__init__(self, controlDes, weights)
    return
