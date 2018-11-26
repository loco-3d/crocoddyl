import abc
import pinocchio as se3


class Integrator(object):
  """ This abstract class allows us to define different integration rules.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __call__(dynamicsModel, dynamicsData, xNext):
    """ Integrate the system dynamics given an user-defined integration scheme.

    :param dynamicsModel: dynamics model
    :param dynamicsData: dynamics data
    :param xNext: next state after the integration
    """
    return NotImplementedError

class EulerIntegrator(Integrator):
  """ Define a forward Euler integrator.
  """
  @staticmethod
  def __call__(dynamicsModel, dynamicsData, xNext):
    """ Integrate the system dynamics using the forward Euler scheme.

    :param dynamicsModel: dynamics model
    :param dynamicsData: dynamics data
    :param xNext: next state after the integration
    """
    xNext[dynamicsModel.nq():] = \
      dynamicsData.x[dynamicsModel.nq():] +\
      dynamicsData.dt * dynamicsData.a
    xNext[:dynamicsModel.nq()] = \
      dynamicsModel.integrateConfiguration(
        dynamicsModel, dynamicsData,
        dynamicsData.x[:dynamicsModel.nq()],
        dynamicsData.dt * xNext[dynamicsModel.nq():])