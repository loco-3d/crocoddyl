import abc
import numpy as np

class DiscretizerBase(object):
  """ This abstract class declares the virtual method for any discretization
  method of system dynamics.
  """
  __metaclass__=abc.ABCMeta
  
  @abc.abstractmethod
  def __init__(self):
    return
  """
  @abc.abstractmethod
  def __call__(ddpModel, ddpIData):
    pass
  """
  class fx(object):
    """ Abstract Class for the derivative for the function wrt x """
    
    __metaclass__=abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, dimf, dimx):
      pass
    """
    @abc.abstractmethod
    def multiply(Vxx, self):
      pass

    @abc.abstractmethod
    def transposemultiply(self, Vxx):
      pass
    """

  class fu(object):
    """ Abstract Class for the derivative for the function wrt u """
    
    __metaclass__=abc.ABCMeta

    @abc.abstractmethod
    def __init__(self, dimf, dimx):
      pass
    """
    @abc.abstractmethod
    def multiply(Vxx, self):
      pass

    @abc.abstractmethod
    def transposemultiply(self, Vxx):
      pass
    """
class FloatingBaseMultibodyEulerDiscretizer(DiscretizerBase):
  """ Convert the time-continuos dynamics into time-discrete one by using
    forward Euler rule."""
  def __init__(self):
    return

  @staticmethod
  def __multiply__(fx, ):
    ddpIData.dynamicsData.fx
    return

  class fx(DiscretizerBase.fx):

    def __init__(self, dimq, dimv, dimf, dimx):
      self.dimq = dimq
      self.dimv = dimv
      self.dimf = dimf
      self.dimx = dimx
      self.dt = 1e-3
      self.aq = np.empty((dimv, dimq)) #derivative of ddq wrt q
      self.av = np.empty((dimv, dimv)) #derivative of ddq wrt v
      self.val = np.empty((dimf, dimf))

    def __call__(self):
      self.val[:self.dimv, :self.dimv] = np.identity(self.dimv)+self.aq*self.dt*self.dt
      self.val[:self.dimv, self.dimv:] = np.identity(self.dimv)*self.dt+self.av*self.dt*self.dt
      self.val[self.dimv:, :self.dimv] = self.aq*self.dt
      self.val[self.dimv:, self.dimv:] = np.identity(self.dimv) + self.av*self.dt
      return self.val

    def premultiply(self,V):
      print "Premultiplying in fx", V
      output = np.empty((V.shape[0], self.dimx))
      output[:,:self.dimv] = V[:, :self.dimv] +\
                             np.dot(V[:, :self.dimv], self.aq)*self.dt+\
                             np.dot(V[:,self.dimv:], self.aq)*self.dt
      output[:,self.dimv:] = V[:, :self.dimv]*self.dt +\
                             np.dot(V[:, :self.dimv], self.av)*self.dt*self.dt +\
                             V[:, self.dimv:] +\
                             np.dot(V[:, self.dimv:], self.av)*self.dt
      return output


    def transposemultiplymat(self, V):
      output = np.empty((self.dimf, V.shape[1]))
      output[:self.dimv, :self.dimv] = \
                              V[:self.dimv, :self.dimv] +\
                              np.dot(self.aq.T, V[:self.dimv, :self.dimv])*self.dt*self.dt +\
                              np.dot(self.aq.T, V[self.dimv:, :self.dimv])*self.dt
      output[:self.dimv, self.dimv:] = \
                              V[:self.dimv, self.dimv:] +\
                              np.dot(self.aq.T, V[:self.dimv, self.dimv:])*self.dt*self.dt +\
                              np.dot(self.aq.T, V[self.dimv:, self.dimv:])*self.dt
      output[self.dimv:, :self.dimv] = \
                              V[:self.dimv, :self.dimv]*self.dt +\
                              np.dot(self.av.T, V[:self.dimv, :self.dimv])*self.dt*self.dt +\
                              V[self.dimv:,:self.dimv] +\
                              np.dot(self.av.T, V[self.dimv:,:self.dimv])*self.dt
      output[self.dimv:,self.dimv:] = \
                              V[:self.dimv, self.dimv:]*self.dt +\
                              np.dot(self.av.T, V[:self.dimv, self.dimv:])*self.dt*self.dt +\
                              V[self.dimv:, self.dimv:] +\
                              np.dot(self.av.T, V[self.dimv:,self.dimv:])*self.dt
      return output

    def transposemultiplyarr(self, V):
      output = np.empty((self.dimf, 1))
      output[:self.dimv, 0] = V[:self.dimv, 0] +\
                              np.dot(self.aq.T, V[:self.dimv, 0])*self.dt*self.dt +\
                              np.dot(self.aq.T, V[self.dimv:, 0])*self.dt
      output[self.dimv:, 0] = V[:self.dimv, 0]*self.dt +\
                              np.dot(self.av.T, V[:self.dimv, 0])*self.dt*self.dt +\
                              V[self.dimv:,0] +\
                              np.dot(self.av.T, V[self.dimv:,0])*self.dt
      return output

  class fu(object):
    def __init__(self, dimv, dimu, dimf):
      self.dimv = dimv
      self.dimu = dimu
      self.dimf = dimf
      #TODO: Load from ddpData
      self.dt = 1e-3
      self.au = np.empty((dimv, dimu)) #derivative of ddq wrt u
      self.val = np.empty((dimf, dimu))
      return

    def premultiply(self,V):
      return np.dot(V[:,:self.dimv], self.au)*self.dt*self.dt+\
        np.dot(V[:,self.dimv:],self.au)*self.dt

    def transposemultiply(self, V_p):
      return np.dot(self.au.T, V_p[:self.dimv,:])*self.dt*self.dt+\
        np.dot(self.au.T, V_p[self.dimv:,:])*self.dt

    def __call__(self):
      self.val[:self.dimv, :] = self.au*self.dt*self.dt
      self.val[self.dimv:, :] = self.au*self.dt
      return self.val

    def square(self):
      au_sq = np.matmul(self.au.T, self.au)
      return au_sq*self.dt*self.dt*(1.+self.dt*self.dt)
