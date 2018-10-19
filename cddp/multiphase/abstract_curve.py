import abc
class AbstractCurve(object):

  __metaclass__=abc.ABCMeta
  def __init__(self, tmin, tmax):
    self.t_min = tmin
    self.t_max = tmax

  @abc.abstractmethod
  def __call__(self, t):
    return NotImplementedError

  def tmin(self):
    return self.t_min

  def tmax(self):
    return self.t_max

  @abc.abstractmethod
  def dim(self):
    pass 
