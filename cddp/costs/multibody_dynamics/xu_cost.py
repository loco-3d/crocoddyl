from cddp.costs.cost_base import QuadraticCostBase
import pinocchio as se3
import numpy as np

class StateCost(QuadraticCostBase):

  def __init__(self, dynamicsModel, stateDes, weights):
    QuadraticCostBase.__init__(self, dynamicsModel, stateDes, weights)
    self.pinocchioModel = dynamicsModel.pinocchioModel
    self.dim = dynamicsModel.nx()

    self._r = np.empty((self.dim, 1))
    self._rx = np.empty((self.dim, dynamicsModel.nx()))
    self._ru = np.empty((self.dim, dynamicsModel.nu()))
    return

  def forwardRunningCalc(self,dynamicsData):
    self._r[:self.dynamicsModel.nv()] = \
                  se3.difference(self.pinocchioModel,
                                 self.ref[:self.dynamicsModel.nq(),:],
                                 dynamicsData.x[:self.dynamicsModel.nq(),:])
    self._r[self.dynamicsModel.nv():] = dynamicsData.x[self.dynamicsModel.nq():,:]- \
                                        self.ref[self.dynamicsModel.nq():,:]

  def forwardTerminalCalc(self,dynamicsData):
    self.forwardRunningCalc(dynamicsData)

  def backwardRunningCalc(self, dynamicsData):
    return

  def backwardTerminalCalc(self, dynamicsData):
    return
  
  def getlu(self):
    return np.zeros((self.dynamicsModel.nu(),1))

  def getlux(self):
    return np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nx()))
  
  def getluu(self):
    return np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nu()))

  def getlgg(self):
    return np.zeros((self.dynamicsModel.dimConstraint, self.dynamicsModel.dimConstraint))

  def getlg(self):
    return np.zeros((self.dynamicsModel.dimConstraint, 1))
    
class ControlCost(QuadraticCostBase):

  def __init__(self, dynamicsModel, controlDes, weights):
    QuadraticCostBase.__init__(self, dynamicsModel, controlDes, weights)
    self.pinocchioModel = dynamicsModel.pinocchioModel
    self.dim = dynamicsModel.nu()

    self._r = np.empty((self.dim, 1))
    self._rx = np.zeros((self.dim, dynamicsModel.nx()))
    self._ru = np.identity(self.dim)
    self._rg = np.identity(self.dim)
    return

  def forwardRunningCalc(self, dynamicsData):
    np.copyto(self._r, dynamicsData.u-self.ref)

  def forwardTerminalCalc(self,dynamicsData):
    self.forwardRunningCalc(dynamicsData)

  def backwardRunningCalc(self, dynamicsData):
    return

  def backwardTerminalCalc(self, dynamicsData):
    return
    
  def getlux(self):
    return np.zeros((self.dynamicsModel.nu(), self.dynamicsModel.nx()))

  def getlx(self):
    return np.zeros((self.dynamicsModel.nx(),1))

  def getlxx(self):
    return np.zeros((self.dynamicsModel.nx(), self.dynamicsModel.nx()))

  def getlgg(self):
    return np.zeros((self.dynamicsModel.dimConstraint, self.dynamicsModel.dimConstraint))

  def getlg(self):
    return np.zeros((self.dynamicsModel.dimConstraint, 1))
