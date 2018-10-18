import abc
import numpy as np

class DDPIntervalDataBase(object):
  """ Define the base class for data element for an interval."""
  __metaclass__=abc.ABCMeta

  def __init__(self, ddpModel):
    self.dynamicsData = ddpModel.dynamicsModel.createIntervalData()
    self.costData = ddpModel.costManager.createIntervalData()
    #self.systemData = ddpModel.createIntervalData()

    self.x = np.empty((ddpModel.dynamicsModel.nx(), 1))
    self.u = np.empty((ddpModel.dynamicsModel.nu(), 1))

    self.t0 = -1.
    self.tf = -1.

  def dynamicsIntegration(self):
    """Performes the dynamics integration to generate the state and control functions"""
    self.dynamicsData.integrate();
    self.costData.stepCost();
  

class TerminalDDPData(DDPIntervalDataBase):
  """ Data structure for the terminal interval of the DDP.

  We create data for the nominal and new state values.
  """
  def __init__(self, ddp_model):
    DDPIntervalDataBase.__init__(self, ddp_model)

    # Nominal and new state on the interval
    #self.x = np.matrix(np.zeros((self.system.nx, 1)))
    #self.x_new = np.matrix(np.zeros((self.system.nx, 1)))

    # Time of the terminal interval
    #self.t = np.matrix(np.zeros(1))
   
class RunningDDPData(DDPIntervalDataBase):
  """ Data structure for the running interval of the DDP.

  We create data for the nominal and new state and control values. Additionally,
  this data structure contains regularized terms too (e.g. Quu_r).
  """
  def __init__(self, ddp_model):
    DDPIntervalDataBase.__init__(self, ddp_model)
    
    # Nominal and new state on the interval
    #self.x = np.matrix(np.zeros((self.system.nx, 1)))
    #self.x_new = np.matrix(np.zeros((self.system.nx, 1)))

    # Nominal and new control command on the interval
    #self.u = np.matrix(np.zeros((self.system.m, 1)))
    #self.u_new = np.matrix(np.zeros((self.system.m, 1)))

    # Starting and final time on the interval
    #self.t0 = np.matrix(np.zeros(1))
    #self.tf = np.matrix(np.zeros(1))

    # Feedback and feedforward terms
    #self.K = np.matrix(np.zeros((self.system.m, self.system.ndx)))
    #self.j = np.matrix(np.zeros((self.system.m, 1)))

    # Value function and its derivatives
    #self.Vx = np.matrix(np.zeros((self.system.ndx, 1)))
    #self.Vxx = np.matrix(np.zeros((self.system.ndx, self.system.ndx)))

    # Quadratic approximation of the value function
    # self.Qx = np.matrix(np.zeros((self.system.ndx, 1)))
    # self.Qu = np.matrix(np.zeros((self.system.m, 1)))
    # self.Qxx = np.matrix(np.zeros((self.system.ndx, self.system.ndx)))
    # self.Qux = np.matrix(np.zeros((self.system.m, self.system.ndx)))
    # self.Quu = np.matrix(np.zeros((self.system.m, self.system.m)))
    # self.Qux_r = np.matrix(np.zeros((self.system.m, self.system.ndx)))
    # self.Quu_r = np.matrix(np.zeros((self.system.m, self.system.m)))
