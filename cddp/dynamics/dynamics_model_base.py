import abc

class DynamicsModelBase():
  "Base class to define the dynamics model"
  __metaclass__=abc.ABCMeta
  @abc.abstractmethod
  def __init__(self):
    pass
  """
  @abc.abstractmethod
  def createData(self):
    pass
  """
