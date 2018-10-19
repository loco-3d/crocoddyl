import abc

class DynamicsModelBase(object):
  "Base class to define the dynamics model"
  __metaclass__=abc.ABCMeta
  @abc.abstractmethod
  def __init__(self):
    pass
  
  @abc.abstractmethod
  def createIntervalData(self):
    pass
  
  @abc.abstractmethod
  def nx(self):
    pass

  @abc.abstractmethod
  def nxImpl(self):
    pass
  
  @abc.abstractmethod
  def nu(self):
    pass
