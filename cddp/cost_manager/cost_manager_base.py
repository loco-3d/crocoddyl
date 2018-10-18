import abc

class CostManagerIntervalDataBase(object):

  __metaclass__=abc.ABCMeta

  @abc.abstractmethod
  def __init__(self):
    pass

  
class CostManagerBase(object):

  __metaclass__=abc.ABCMeta

  @abc.abstractmethod
  def __init__(self):
    pass

  @abc.abstractmethod
  def createIntervalData(self):
    return CostManagerIntervalDataBase(self)
