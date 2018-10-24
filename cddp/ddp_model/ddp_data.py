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
    self.intervalDataVector = [RunningDDPData(ddp_model, timeline[i], timeline[i+1])
                               for i in xrange(self.N)]
    self.intervalDataVector.append(TerminalDDPData(ddp_model, timeline[-1]))

    #Run time variables
    self._convergence = False
    self.muLM = -1.
    self.muV = -1.
    self.alpha = -1.
    self.n_iter = -1.
