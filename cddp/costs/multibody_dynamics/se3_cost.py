from cost_base import QuadraticCostBase

class SE3Cost(QuadraticCostBase):

  def __init__(self, Mdes, weight, frame_name):
    QuadraticCostBase.__init__(self, Mdes, weight)
    self.frame_name = frame_name
    return
