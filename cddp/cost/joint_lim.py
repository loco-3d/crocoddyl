from cddp.cost.soft_constraint import XLogBarrier
import pinocchio as se3
import numpy as np


class JointPositionBarrier(XLogBarrier):
  """ Log-barrier for joint position limits of the robot.
  """

  def __init__(self, model):
    """ Construct the barrier function for the joint position limits.

    :param model: Pinocchio model
    :param lb: lower bound
    """
    self._model = model
    self._n_joint = self._model.nq

    ub = self._model.upperPositionLimit
    lb = self._model.lowerPositionLimit
    bound = np.vstack([ub, -lb])
    XLogBarrier.__init__(self, bound)

  def r(self, system, data, x, u):
    q = x[:self._model.nq]
    data.r[:self._n_joint] = q
    data.r[self._n_joint:] = -q
    return data.r

  def rx(self, system, data, x, u):
    data.rx[:self._n_joint,:self._n_joint] = np.ones((self._n_joint, self._n_joint))
    data.rx[self._n_joint:,:self._n_joint] = -np.ones((self._n_joint, self._n_joint))
    return data.rx