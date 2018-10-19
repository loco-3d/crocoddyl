import abc
import pinocchio as se3

class IntegratorBase(object):
  """ Class to integrate the dynamics. Does not store data.
  """
  __metaclass__=abc.ABCMeta

  @abc.abstractmethod
  def __init__(self):
    pass

  @abc.abstractmethod
  def __call__(ddpModel, ddpIData, xNext):
    return NotImplementedError

class FloatingBaseMultibodyEulerIntegrator(IntegratorBase):
  """ Create the internal data of forward Euler integrator of the system
  dynamics.
  """

  def __init__(self):
    return
  
  @staticmethod
  def __call__(ddpModel, ddpIData, xNext):
    xNext[ddpModel.dynamicsModel.nq():] = \
              ddpIData.dynamicsData.x[ddpModel.dynamicsModel.nq():] \
              + ddpIData.dynamicsData.pinocchioData.ddq * ddpIData.dt
    xNext[:ddpModel.dynamicsModel.nq()] = \
              se3.integrate(ddpModel.dynamicsModel.pinocchioModel,
                            ddpIData.dynamicsData.x[:ddpModel.dynamicsModel.nq()],
                            xNext[ddpModel.dynamicsModel.nq():] * ddpIData.dt )
