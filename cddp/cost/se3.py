from cddp.cost import RunningResidualQuadraticCost
import numpy as np
import pinocchio as se3
import math


class SE3RunningCost(RunningResidualQuadraticCost):
  def __init__(self, robot, ee_frame, M_des):
    self.robot = robot
    self._frame_idx = self.robot.model.getFrameId(ee_frame)
    self.M_des = M_des
    RunningResidualQuadraticCost.__init__(self, 6)

  def r(self, data, x, u):
    q = x[:self.robot.nq]
    np.copyto(data.r,
      se3.log(self.M_des.inverse() * self.robot.framePosition(q, self._frame_idx)).vector)
    return data.r

  def rx(self, data, x, u):
    q = x[:self.robot.nq]
    data.rx[:, :self.robot.nq] = \
      se3.jacobian(self.robot.model, self.robot.data, q,
                   self.robot.model.frames[self._frame_idx].parent, False, True)
    return data.rx

  def ru(self, data, x, u):
    return data.ru