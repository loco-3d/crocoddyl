from cddp.cost.soft_constraint import XLogBarrier
import pinocchio as se3
import numpy as np


class JointPositionBarrier(XLogBarrier):
  """ Log-barrier for joint position limits of the robot.
  """

  def __init__(self, robot):
    """ Construct the barrier function for the joint position limits.

    :param robot: Pinocchio robot model
    :param lb: lower bound
    """
    self.robot = robot
    self._n_joint = self.robot.nq

    ub = self.robot.model.upperPositionLimit
    lb = self.robot.model.lowerPositionLimit
    bound = np.vstack([ub, -lb])
    XLogBarrier.__init__(self, bound)

  def r(self, data, x, u):
    q = x[:self.robot.nq]
    data.r[:self._n_joint] = q
    data.r[self._n_joint:] = -q
    return data.r

  def rx(self, data, x, u):
    data.rx[:self._n_joint,:self._n_joint] = np.ones((self._n_joint, self._n_joint))
    data.rx[self._n_joint:,:self._n_joint] = -np.ones((self._n_joint, self._n_joint))
    return data.rx