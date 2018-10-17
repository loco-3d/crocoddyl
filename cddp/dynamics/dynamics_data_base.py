import abc

class DynamicsDataBase(metaclass=abc.ABCMeta):
  "Base class to define interface for Dynamics"

  def __init__(self, ddp_model):
    pass

  @abc.AbstractMethod
  def computeAllTerms(self):
    "implement compute all terms"
    pass
