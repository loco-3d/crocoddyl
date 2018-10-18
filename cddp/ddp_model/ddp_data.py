from ddp_interval_data import RunningDDPData, TerminalDDPData
from ddp_model import DDPModel

class DDPData(object):
  """ Base class to define the structure for storing and accessing data elements at each 
  DDP interval
  """

  def __init__(self, ddp_model, timeline):
    """Initializing the data elements"""

    self.timeline = timeline
    self.ddpModel = ddp_model
    self.N = len(timeline) - 1
    self.intervals = [RunningDDPData(ddp_model) for i in xrange(self.N)]
    self.intervals.append(TerminalDDPData(ddp_model))
