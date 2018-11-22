import abc
import pinocchio as se3

class Integrator(object):
  """ This abstract class allows us to define different integration rules.
  """
  __metaclass__=abc.ABCMeta

  @abc.abstractmethod
  def __call__(ddpModel, ddpIData, xNext):
    """ Integrate the system dynamics given an user-defined integration scheme.
    """
    return NotImplementedError

class EulerIntegrator(Integrator):
  """ Define a forward Euler integrator.
  """
  @staticmethod
  def __call__(ddpModel, ddpIData, xNext):
    """ Integrate the system dynamics using the forward Euler scheme.
    """
    xNext[ddpModel.dynamicsModel.nq():] = \
      ddpIData.dynamicsData.x[ddpModel.dynamicsModel.nq():] +\
      ddpIData.dt * ddpIData.dynamicsData.a
    xNext[:ddpModel.dynamicsModel.nq()] = \
      ddpModel.dynamicsModel.integrateConfiguration(
        ddpModel.dynamicsModel,
        ddpIData.dynamicsData.x[:ddpModel.dynamicsModel.nq()],
        ddpIData.dt * xNext[ddpModel.dynamicsModel.nq():])