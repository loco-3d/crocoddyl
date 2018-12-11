import abc
import numpy as np


class DiscretizerData(object):
  """ This class describes the common data for each discretization rule.
  """
  __metaclass__ = abc.ABCMeta
  def __init__(self, dynamicModel):
    self.fx = np.zeros((dynamicModel.nx(), dynamicModel.nx()))
    self.fu = np.zeros((dynamicModel.nx(), dynamicModel.nu()))


class Discretizer(object):
  """ This abstract class allows us to define different discretization rules.

  A discretizer converts the time-continuos dynamics into time-discrete one.
  """
  __metaclass__=abc.ABCMeta
  @abc.abstractmethod
  def __init__(self, dynamicModel):
    return

  @abc.abstractmethod
  def createData(self, dynamicModel, dt):
    """ Create the discretizer data.

    :param dynamicModel: dynamics model
    :param dt: step integration
    """
    pass

  @abc.abstractmethod
  def __call__(dynamicModel, dynamicsData):
    """ Discretize the system dynamics given an user-defined discretization scheme.

    :param dynamicModel: dynamics model
    :param dynamicsData: dynamics data
    """
    pass

  # class fx(object):
  #   """ Abstract Class for the derivative for the function wrt x """    
  #   __metaclass__=abc.ABCMeta
  #   @abc.abstractmethod
  #   def __init__(self, dimx, dt):
  #     pass
  #   @abc.abstractmethod
  #   def backwardRunningCalc(self, dynamicModel, dynamicsData):
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
  #   def backwardRunningCalc(self, dynamicModel, dynamicsData):
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
  def __init__(self, dynamicModel, dt):
    DiscretizerData.__init__(self, dynamicModel)

    # Extra terms for the simpletic Euler discretizer
    self.dt2 = dt * dt
    self.I = np.identity(dynamicModel.nv())
    self.dt_I = dt * np.identity(dynamicModel.nv())


class EulerDiscretizer(Discretizer):
  """ Define a forward Euler discretizer.
  """
  def __init__(self):
    return

  def createData(self, dynamicModel, dt):
    """ Create the Euler discretizer data.

    :param dynamicModel: dynamics model
    :param dt: step integration
    """
    return EulerDiscretizerData(dynamicModel, dt)

  @staticmethod
  def __call__(dynamicModel, dynamicsData):
    """ Discretize the system dynamics using the forward Euler scheme.

    :param dynamicModel: dynamics model
    :param dynamicsData: dynamics data
    """
    # Discretizing fx
    dynamicsData.discretizer.fx[:dynamicModel.nv(),:dynamicModel.nv()] = \
      dynamicsData.discretizer.I + dynamicsData.discretizer.dt2 * dynamicsData.aq
    dynamicsData.discretizer.fx[:dynamicModel.nv(),dynamicModel.nv():] = \
      dynamicsData.discretizer.dt_I + dynamicsData.discretizer.dt2 * dynamicsData.av
    dynamicsData.discretizer.fx[dynamicModel.nv():,:dynamicModel.nv()] = \
      dynamicsData.dt * dynamicsData.aq
    dynamicsData.discretizer.fx[dynamicModel.nv():,dynamicModel.nv():] = \
      dynamicsData.discretizer.I + dynamicsData.dt * dynamicsData.av

    # Discretizing fu
    dynamicsData.discretizer.fu[:dynamicModel.nv(),:] = \
      dynamicsData.discretizer.dt2 * dynamicsData.au
    dynamicsData.discretizer.fu[dynamicModel.nv():,:] = \
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

  #   def backwardRunningCalc(self, dynamicModel, dynamicsData):
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

  #   def backwardRunningCalc(self, dynamicModel, dynamicsData):
  #     self.au = dynamicsData.au
  #     self.val[self.dimv:, :] = self.au*self.dt
  #     self.fu_sq = np.dot(self.au.T, self.au)*self.dt*self.dt
  #     return

  #   def square(self):
  #     return self.fu_sq