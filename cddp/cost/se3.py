from cddp.cost import RunningResidualQuadraticCost
import numpy as np
import pinocchio as se3
import math


class SE3RunningCost(RunningResidualQuadraticCost):
  """ Running cost given a desired SE3 pose.

  This cost function can be only defined for geometric systems. For these 
  systems, the state x=(q,v) contains the configuration point q\in Q and its the
  tangent velocity v\in TxQ. The dimension of the configuration space Q, its
  tangent space TxQ and the control (input) vector are nq, nv and m,
  respectively.
  """
  def __init__(self, robot, ee_frame, M_des):
    """ Construct the running cost for a desired SE3.

    It requires the robot model created by Pinocchio, the frame name and its
    desired SE3 pose.
    :param robot: Pinocchio robot model
    :ee_frame: frame name
    :M_des: desired SE3 pose
    """
    self.robot = robot
    self._frame_idx = self.robot.model.getFrameId(ee_frame)
    self.M_des = M_des
    # Residual vector lies in the tangent space of the SE3 manifold
    RunningResidualQuadraticCost.__init__(self, 6)

  def r(self, data, x, u):
    """ Compute SE3 error vector and store the result in data.

    The SE3 error is mapped in the tangent space of the SE3 manifold. We
    computed the frame SE3 pose given a state x. Then we map this to its tangent
    motion through the log function.
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: SE3 error
    """
    # SE3 error mapping in its tangent manifold
    q = x[:self.robot.nq]
    np.copyto(data.r,
      se3.log(self.M_des.inverse() * self.robot.framePosition(q, self._frame_idx)).vector)
    return data.r

  def rx(self, data, x, u):
    """ Compute the state Jacobian of the SE3 error vector and store the result
    in data.

    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: state Jacobian of the SE3 error vector
    """
    q = x[:self.robot.nq]
    data.rx[:, :self.robot.nv] = \
      se3.jointJacobian(self.robot.model, self.robot.data, q,
                   self.robot.model.frames[self._frame_idx].parent,
                   se3.ReferenceFrame.LOCAL, True)
    return data.rx

  def ru(self, data, x, u):
    """ Compute the control Jacobian of the SE3 error vector and store the result
    in data.

    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: control Jacobian of the SE3 error vector
    """
    return data.ru