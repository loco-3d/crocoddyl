import abc
import numpy as np

class CostBase(object):
  """Base class for defining costs."""

  __metaclass__=abc.ABCMeta
  
  def __init__(self, dynamicsModel, ref, weight):
    self.dynamicsModel = dynamicsModel
    self.ref = ref
    self.weight = weight
    self.dim = -1

    self.l = 0.
    pass

  @abc.abstractmethod
  def forwardRunningCalc(self, dynamicsData):
    pass

class QuadraticCostBase(CostBase):
  """This abstract class creates a quadratic cost of the form:
  0.5 xr^T Q xr.

  An important remark here is that the state residual (i.e. xr) depends linearly
  on the state. This cost function can be used to define a goal state penalty.
  This residual has to be implemented in a derived class (i.e. xr function).

  Before computing the cost values, it is needed to set up the Q matrix. We
  define it as diagonal matrix because: 1) it is easy to tune and 2) we can
  exploit an efficient computation of it. Additionally, for efficiency
  computation, we assume that compute first l (or lx) whenever you want
  to compute lx (or lxx)."""

  __metaclass__=abc.ABCMeta

  def __init__(self,dynamicsModel, ref,weight):
    CostBase.__init__(self, dynamicsModel, ref,weight)
    pass

  def getl(self):
    # The quadratic term is as follows 0.5 * xr^T Q xr. We compute it
    # efficiently by exploiting the fact thar Q is a diagonal matrix
    return 0.5 * np.asscalar(np.dot(self._r.T, np.multiply(self.weight, self._r)))
