import abc

class IntegratorBase(object):
  """ Class to integrate the dynamics. Does not store data.
  """
  __metaclass__=abc.ABCMeta
  def __init__(self, dynamicsModel):
    self.dynamicsModel = dynamicsModel
    return
  """
  @abc.abstractmethod
  @staticmethod
  def integrate(self, systemData, dynamicsData):
    return NotImplementedError
  """
class EulerIntegrator(IntegratorBase):
  """ Create the internal data of forward Euler integrator of the system
  dynamics.

  """
  def __init__(self, dynamicsModel):
    IntegratorBase.__init__(self,dynamicsModel)
    return
