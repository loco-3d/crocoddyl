import abc

class CostBase(object):
  """Base class for defining costs dependent on multibody dynamics"""

  __metaclass__=abc.ABCMeta
  
  @abc.abstractmethod
  def __init__(self, ref, weight):
    self.ref = ref
    self.weight = weight
    pass

class QuadraticCostBase(CostBase):
  """This abstract class creates a quadratic terminal cost of the form:
  0.5 xr^T Q xr.

  An important remark here is that the state residual (i.e. xr) depends linearly
  on the state. This cost function can be used to define a goal state penalty.
  This residual has to be implemented in a derived class (i.e. xr function).

  Before computing the cost values, it is needed to set up the Q matrix. We
  define it as diagonal matrix because: 1) it is easy to tune and 2) we can
  exploit an efficient computation of it. Additionally, for efficiency
  computation, we assume that compute first l (or lx) whenever you want
  to compute lx (or lxx)."""

  @abc.abstractmethod
  def __init__(self,ref,weight):
    CostBase.__init__(self, ref,weight)
    pass
