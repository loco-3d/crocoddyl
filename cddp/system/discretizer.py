import abc

class DiscretizerBase(object):
  """ This abstract class declares the virtual method for any discretization
  method of system dynamics.
  """
  __metaclass__=abc.ABCMeta

  @abc.abstractmethod
  def __init__(self):
    return

  @abc.abstractmethod
  def __call__(ddpModel, ddpIData):
    pass

class FloatingBaseMultibodyEulerDiscretizer(DiscretizerBase):
  """ Convert the time-continuos dynamics into time-discrete one by using
    forward Euler rule."""

  def __init__(self):
    return

  @staticmethod
  def __call__(ddpModel, ddpIData):
    pass
