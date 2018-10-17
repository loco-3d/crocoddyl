import abc

class DiscretizerBase(object):
  """ This abstract class declares the virtual method for any discretization
  method of system dynamics.
  """
  __metaclass__=abc.ABCMeta
  def __init__(self, dynamicsModel):
    self.dynamicsModel = dynamicsModel
    return
  """
  @abc.abstractmethod
  @staticmethod
  def nOrderApproximate(self, systemData, dynamicsData):
    pass
  """
class EulerDiscretizer(DiscretizerBase):

  def __init__(self, dynamicsModel):
    DiscretizerBase.__init__(self, dynamicsModel)
    """
    def __call__(self, dynamicsModel):
     Convert the time-continuos dynamics into time-discrete one by using
    forward Euler rule.

    :param system: system model
    :param data: system data
    :param x: state vector
    :param u: control vector
    :param dt: sampling period
    :returns: discrete time state and control derivatives
    system.fx(data, x, u)
    system.fu(data, x, u)
    np.copyto(data.fx, self._I + data.fx * dt)
    np.copyto(data.fu, data.fu * dt)
    return data.fx, data.fu
    """
    return
