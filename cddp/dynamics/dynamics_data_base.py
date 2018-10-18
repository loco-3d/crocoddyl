import abc

class DynamicsDataBase(object):
  "Base class to define interface for Dynamics"
  __metaclass__=abc.ABCMeta

  @abc.abstractmethod  
  def __init__(self, ddp_model):
    pass

  @abc.abstractmethod
  def computeAllTerms(self):
    "implement compute all terms"
    pass
