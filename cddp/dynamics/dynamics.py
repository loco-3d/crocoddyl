import abc
import numpy as np


class DynamicsData(object):
  "Base class to define interface for Dynamics"
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, ddpModel):
    self.diff_x = np.zeros((ddpModel.dynamicsModel.nx(), 1))

    # Current and previous state and control
    self.x = np.zeros((ddpModel.dynamicsModel.nxImpl(), 1))
    self.u = np.zeros((ddpModel.dynamicsModel.nu(), 1))
    self.x_prev = np.zeros((ddpModel.dynamicsModel.nxImpl(), 1))
    self.u_prev = np.zeros((ddpModel.dynamicsModel.nu(), 1))

class DynamicsModel(object):
  "Base class to define the dynamics model"
  __metaclass__ = abc.ABCMeta
  
  def __init__(self, nxImpl, nx, nu):
    self._nx_impl = nxImpl
    self._nx = nx
    self._nu = nu

  @staticmethod
  def forwardRunningCalc(dynamicsModel, dynamicsData):
    "implement compute all terms for forward pass"
    pass

  @staticmethod
  def forwardTerminalCalc(dynamicsModel, dynamicsData):
    "implement compute all terms for forward pass"
    pass

  @staticmethod
  def backwardRunningCalc(dynamicsModel, dynamicsData):
    "implement compute all terms for backward pass"
    pass

  @staticmethod
  def backwardTerminalCalc(dynamicsModel, dynamicsData):
    "implement compute all terms for backward pass"
    pass

  @abc.abstractmethod
  def createData(self):
    pass

  @staticmethod
  def deltaX(dynamicsModel, dynamicsData, x0, x1):
    pass

  def nxImpl(self):
    return self._nx_impl

  def nx(self):
    return self._nx

  def nu(self):
    return self._nu