from ddp_interval_data import RunningDDPData, TerminalDDPData
from ddp_model import DDPModel
import numpy as np

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

    #Total Cost
    self.totalCost = 0.
    self.totalCost_prev = 0.
    self.dV_exp = 0.
    self.dV = 0.
    
    #Run time variables
    self._convergence = False
    self.muLM = -1.
    self.muV = -1.
    self.muI = np.zeros((self.ddpModel.dynamicsModel.nu(),self.ddpModel.dynamicsModel.nu()))
    self.alpha = -1.
    self.n_iter = -1.
    
    #Analysis Variables
    self.gamma = 0.
    self.theta = 0.
    self.z_new = 0.
    self.z = 0.
