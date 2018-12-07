import abc
import pinocchio as se3


class Integrator(object):
  """ This abstract class allows us to define different integration rules.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __call__(dynamicsModel, dynamicsData, x, u, xNext):
    """ Integrate the system dynamics given an user-defined integration scheme.

    :param dynamicsModel: dynamics model
    :param dynamicsData: dynamics data
    :param x: current state
    :param u: current control
    :param xNext: next state after the integration
    """
    raise NotImplementedError("Not implemented yet.")

class EulerIntegrator(Integrator):
  """ Define a forward Euler integrator.
  """
  @staticmethod
  def __call__(dynamicsModel, dynamicsData, x, u, xNext):
    """ Integrate the system dynamics using the forward Euler scheme.

    :param dynamicsModel: dynamics model
    :param dynamicsData: dynamics data
    :param x: current state
    :param u: current control
    :param xNext: next state after the integration
    """
    # Updating the dynamics
    dynamicsModel.updateDynamics(dynamicsData, x, u)

    xNext[dynamicsModel.nq():] = \
      x[dynamicsModel.nq():] +\
      dynamicsData.dt * dynamicsData.a
    xNext[:dynamicsModel.nq()] = \
      dynamicsModel.integrateConfiguration(
        x[:dynamicsModel.nq()],
        dynamicsData.dt * xNext[dynamicsModel.nq():])