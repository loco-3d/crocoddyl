import abc
import numpy as np


class DiscretizerData(object):
  __metaclass__ = abc.ABCMeta
  def __init__(self, dynamicsModel):
    self.fx = np.zeros((dynamicsModel.nx(), dynamicsModel.nx()))
    self.fu = np.zeros((dynamicsModel.nx(), dynamicsModel.nu()))


class Discretizer(object):
  """ This abstract class declares the virtual method for any discretization
  method of system dynamics.
  """
  __metaclass__=abc.ABCMeta
  @abc.abstractmethod
  def __init__(self, dynamicsModel):
    return

  @abc.abstractmethod
  def createData(self, dynamicsModel, dt):
    pass

  @abc.abstractmethod
  def __call__(dynamicsModel, dynamicsData):
    pass

  # class fx(object):
  #   """ Abstract Class for the derivative for the function wrt x """    
  #   __metaclass__=abc.ABCMeta
  #   @abc.abstractmethod
  #   def __init__(self, dimx, dt):
  #     pass
  #   @abc.abstractmethod
  #   def backwardRunningCalc(self, dynamicsModel, dynamicsData):
  #     pass
  #   @abc.abstractmethod
  #   def __call__(self):
  #     pass
  #   @abc.abstractmethod
  #   def premultiply(self,V, output):
  #     pass
  #   @abc.abstractmethod
  #   def transposemultiplymat(self, V):
  #     pass
  #   @abc.abstractmethod
  #   def transposemultiplyarr(self, V):
  #     pass

  # class fu(object):
  #   """ Abstract Class for the derivative for the function wrt u """
  #   __metaclass__=abc.ABCMeta
  #   @abc.abstractmethod
  #   def __init__(self, dimx, dimu, dt):
  #     pass
  #   @abc.abstractmethod
  #   def backwardRunningCalc(self, dynamicsModel, dynamicsData):
  #     pass
  #   @abc.abstractmethod
  #   def __call__(self):
  #     pass
  #   @abc.abstractmethod
  #   def premultiply(self,V, output):
  #     pass
  #   @abc.abstractmethod
  #   def transposemultiplyarr(self, V):
  #     pass
  #   @abc.abstractmethod
  #   def transposemultiplymat(self, V):
  #     pass
  #   @abc.abstractmethod
  #   def square(self):
  #     pass


class EulerDiscretizerData(DiscretizerData):
  def __init__(self, dynamicsModel, dt):
    DiscretizerData.__init__(self, dynamicsModel)
    # Update once the upper-block
    self.I = np.identity(dynamicsModel.nv())
    self.fx[:dynamicsModel.nv(),:dynamicsModel.nv()] = self.I
    self.fx[:dynamicsModel.nv(),dynamicsModel.nv():] = dt * self.I


class EulerDiscretizer(Discretizer):
  """ Convert the time-continuos dynamics into time-discrete one by using
    forward Euler rule."""
  def __init__(self):
    return

  def createData(self, dynamicsModel, dt):
    return EulerDiscretizerData(dynamicsModel, dt)

  @staticmethod
  def __call__(dynamicsModel, dynamicsData):
    dynamicsData.discretizer.fx[dynamicsModel.nv():,:dynamicsModel.nv()] = \
      dynamicsData.dt * dynamicsData.aq
    dynamicsData.discretizer.fx[dynamicsModel.nv():,dynamicsModel.nv():] = \
      dynamicsData.discretizer.I + dynamicsData.dt * dynamicsData.av
    dynamicsData.discretizer.fu[dynamicsModel.nv():,:] = \
      dynamicsData.dt * dynamicsData.au

  # class fx(DiscretizerBase.fx):
  #   def __init__(self, dimx, dt):
  #     self.dimv = dimx/2
  #     self.dimx = dimx
  #     self.dt = dt
  #     #Const ref to dynamicsData aq
  #     self.aq = None #derivative of ddq wrt q
  #     #Const ref to dynamicsData av
  #     self.av = None #np.zeros((dimv, dimv)) #derivative of ddq wrt v

  #     self.val = np.zeros((dimx, dimx))

  #     #Runtime Values
  #     self.outputarr = np.zeros((dimx, 1))
  #     self.outputmat = np.zeros((dimx, dimx))

  #   def backwardRunningCalc(self, dynamicsModel, dynamicsData):
  #     self.aq = dynamicsData.aq
  #     self.av = dynamicsData.av
  #     self.val[:self.dimv, :self.dimv] = np.identity(self.dimv)
  #     self.val[:self.dimv, self.dimv:] = np.identity(self.dimv)*self.dt
  #     self.val[self.dimv:, :self.dimv] = self.aq*self.dt
  #     self.val[self.dimv:, self.dimv:] = np.identity(self.dimv) + self.av*self.dt
  #     return

  #   def __call__(self):
  #     return self.val

  #   def premultiply(self,V, output):
  #     np.copyto(output, self.transposemultiplymat(V.T).T)
  #     return

  #   def transposemultiplymat(self, V):
  #     self.outputmat[:self.dimv, :self.dimv] = \
  #                             V[:self.dimv, :self.dimv] +\
  #                             np.dot(self.aq.T, V[self.dimv:, :self.dimv])*self.dt
  #     self.outputmat[:self.dimv, self.dimv:] = \
  #                             V[:self.dimv, self.dimv:] +\
  #                             np.dot(self.aq.T, V[self.dimv:, self.dimv:])*self.dt
  #     self.outputmat[self.dimv:, :self.dimv] = \
  #                             V[:self.dimv, :self.dimv]*self.dt +\
  #                             V[self.dimv:,:self.dimv] +\
  #                             np.dot(self.av.T, V[self.dimv:,:self.dimv])*self.dt
  #     self.outputmat[self.dimv:,self.dimv:] = \
  #                             V[:self.dimv, self.dimv:]*self.dt +\
  #                             V[self.dimv:, self.dimv:] +\
  #                             np.dot(self.av.T, V[self.dimv:,self.dimv:])*self.dt
  #     return self.outputmat

  #   def transposemultiplyarr(self, V):
  #     self.outputarr[:self.dimv, 0] = V[:self.dimv, 0] +\
  #                             np.dot(self.aq.T, V[self.dimv:, 0])*self.dt
  #     self.outputarr[self.dimv:, 0] = V[:self.dimv, 0]*self.dt +\
  #                             V[self.dimv:,0] +\
  #                             np.dot(self.av.T, V[self.dimv:,0])*self.dt
  #     return self.outputarr

  # class fu(object):
  #   def __init__(self, dimx, dimu, dt):
  #     self.dimv = dimx/2
  #     self.dimu = dimu
  #     self.dimx = dimx
  #     #TODO: Load from ddpData
  #     self.dt = dt
  #     #Const ref to dynamicsmodel
  #     self.au = None # np.zeros((dimv, dimu)) #derivative of ddq wrt 

  #     self.val = np.zeros((dimx, dimu))
  #     self.fu_sq = np.zeros((dimu, dimu))

  #     #Runtime Variables
  #     self.outputmat = np.zeros((dimu, dimx))
  #     self.outputarr = np.zeros((dimu, 1))
  #     return

  #   def __call__(self):
  #     return self.val

  #   def premultiply(self,V, output):
  #     np.copyto(output, self.transposemultiplymat(V.T).T)

  #   def transposemultiplyarr(self, V_p):
  #     self.outputarr = np.dot(self.au.T, V_p[self.dimv:,:])*self.dt
  #     return self.outputarr
    
  #   def transposemultiplymat(self, V_p):
  #     self.outputmat = np.dot(self.au.T, V_p[self.dimv:,:])*self.dt
  #     return self.outputmat

  #   def backwardRunningCalc(self, dynamicsModel, dynamicsData):
  #     self.au = dynamicsData.au
  #     self.val[self.dimv:, :] = self.au*self.dt
  #     self.fu_sq = np.dot(self.au.T, self.au)*self.dt*self.dt
  #     return

  #   def square(self):
  #     return self.fu_sq