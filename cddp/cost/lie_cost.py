from cddp.cost import RunningResidualQuadraticCost
import numpy as np
import pinocchio as se3
import math


class SE3RunningCost(RunningResidualQuadraticCost):
  """ Running cost given a desired SE3 pose of a given frame.

  This cost function can be only defined for geometric systems. For these 
  systems, the state x=(q,v) contains the configuration point q\in Q and its the
  tangent velocity v\in TxQ. The dimension of the configuration space Q, its
  tangent space TxQ and the control (input) vector are nq, nv and m,
  respectively.
  """
  def __init__(self, model, ee_frame, M_des):
    """ Construct the running cost for a desired SE3 of a given frame.

    It requires the robot model created by Pinocchio, the frame name and its
    desired SE3 pose.
    :param model: Pinocchio model
    :ee_frame: frame name
    :M_des: desired SE3 pose
    """
    self._model = model
    self._data = self._model.createData()
    self._frame_idx = self._model.getFrameId(ee_frame)
    self.M_des = M_des
    # Residual vector lies in the tangent space of the SE3 manifold
    RunningResidualQuadraticCost.__init__(self, 6)

  def r(self, system, data, x, u):
    """ Compute SE3 error vector and store the result in data.

    The SE3 error is mapped in the tangent space of the SE3 manifold. We
    computed the frame SE3 pose given a state x. Then we map this to its tangent
    motion through the log function.
    :param system: system
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: SE3 error
    """
    # Computing the frame position
    q = x[:self._model.nq]
    se3.forwardKinematics(self._model, self._data, q)
    frame = self._model.frames[self._frame_idx]
    oMf = self._data.oMi[frame.parent].act(frame.placement)

    # SE3 error mapping in its tangent manifold
    np.copyto(data.r, se3.log(self.M_des.inverse() * oMf).vector)
    return data.r

  def rx(self, system, data, x, u):
    """ Compute the state Jacobian of the SE3 error vector and store the result
    in data.

    :param system: system
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: state Jacobian of the SE3 error vector
    """
    q = x[:self._model.nq]
    data.rx[:, :self._model.nv] = \
      se3.jointJacobian(self._model, self._data, q,
                   self._model.frames[self._frame_idx].parent,
                   se3.ReferenceFrame.LOCAL, True)
    return data.rx

  def ru(self, system, data, x, u):
    """ Compute the control Jacobian of the SE3 error vector and store the
    result in data.

    :param system: system
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: control Jacobian of the SE3 error vector
    """
    return data.ru


class TranslationRunningCost(RunningResidualQuadraticCost):
  """ Running cost given a desired Euclidean position for a given frame.

  This cost function can be only defined for geometric systems. For these 
  systems, the state x=(q,v) contains the configuration point q\in Q and its the
  tangent velocity v\in TxQ. The dimension of the configuration space Q, its
  tangent space TxQ and the control (input) vector are nq, nv and m,
  respectively.
  """
  def __init__(self, model, ee_frame, t_des):
    """ Construct the running cost for a desired Euclidean position of a
    given frame.

    It requires the robot model created by Pinocchio, the frame name and its
    desired Euclieand position.
    :param model: Pinocchio model
    :ee_frame: frame name
    :M_des: desired SE3 pose
    """
    self._model = model
    self._data = self._model.createData()
    self._frame_idx = self._model.getFrameId(ee_frame)
    self.t_des = t_des
    self.M_des = se3.SE3(np.eye(3), self.t_des)
    # Residual vector lies in the tangent space of the translation manifold T
    RunningResidualQuadraticCost.__init__(self, 3)

  def r(self, system, data, x, u):
    """ Compute translational error vector and store the result in data.

    :param system: system
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: translational error
    """
    # Computing the frame position
    q = x[:self._model.nq]
    se3.forwardKinematics(self._model, self._data, q)
    frame = self._model.frames[self._frame_idx]
    oMf = self._data.oMi[frame.parent].act(frame.placement)

    # T error mapping in its tangent manifold
    self.M_des.rotation = oMf.rotation
    np.copyto(data.r, se3.log(self.M_des.inverse() * oMf).vector[:3])
    return data.r

  def rx(self, system, data, x, u):
    """ Compute the state Jacobian of the translational error vector and
    store the result in data.

    :param system: system
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: state Jacobian of the SE3 error vector
    """
    q = x[:self._model.nq]
    data.rx[:, :self._model.nv] = \
      se3.jointJacobian(self._model, self._data, q,
                   self._model.frames[self._frame_idx].parent,
                   se3.ReferenceFrame.LOCAL, True)[:3,:]
    return data.rx

  def ru(self, system, data, x, u):
    """ Compute the control Jacobian of the translational error vector and
    store the result in data.

    :param system: system
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: control Jacobian of the SE3 error vector
    """
    return data.ru



class SO3RunningCost(RunningResidualQuadraticCost):
  """ Running cost given a desired SO3 orientaiton of a given frame.

  This cost function can be only defined for geometric systems. For these 
  systems, the state x=(q,v) contains the configuration point q\in Q and its the
  tangent velocity v\in TxQ. The dimension of the configuration space Q, its
  tangent space TxQ and the control (input) vector are nq, nv and m,
  respectively.
  """
  def __init__(self, model, ee_frame, R_des):
    """ Construct the running cost for a desired SO3 of a given frame.

    It requires the robot model created by Pinocchio, the frame name and its
    desired SO3 orientation.
    :param model: Pinocchio model
    :ee_frame: frame name
    :R_des: desired SO3 orientation
    """
    self._model = model
    self._data = self._model.createData()
    self._frame_idx = self._model.getFrameId(ee_frame)
    self.R_des = R_des
    # Residual vector lies in the tangent space of the SO3 manifold
    RunningResidualQuadraticCost.__init__(self, 3)

  def r(self, system, data, x, u):
    """ Compute SO3 error vector and store the result in data.

    The SO3 error is mapped in the tangent space of the SO3 manifold. We
    computed the frame SO3 orientation given a state x. Then we map this to its
    tangent motion through the log function.
    :param system: system
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: SO3 error
    """
    # Computing the frame position
    q = x[:self._model.nq]
    se3.forwardKinematics(self._model, self._data, q)
    frame = self._model.frames[self._frame_idx]
    oRf = self._data.oMi[frame.parent].act(frame.placement).rotation

    # SE3 error mapping in its tangent manifold
    np.copyto(data.r, se3.log3(np.linalg.inv(self.R_des) * oRf))
    return data.r

  def rx(self, system, data, x, u):
    """ Compute the state Jacobian of the SO3 error vector and store the result
    in data.

    :param system: system
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: state Jacobian of the SO3 error vector
    """
    q = x[:self._model.nq]
    data.rx[:, :self._model.nv] = \
      se3.jointJacobian(self._model, self._data, q,
                   self._model.frames[self._frame_idx].parent,
                   se3.ReferenceFrame.LOCAL, True)[3:,:]
    return data.rx

  def ru(self, system, data, x, u):
    """ Compute the control Jacobian of the SE3 error vector and store the
    result in data.

    :param system: system
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: control Jacobian of the SE3 error vector
    """
    return data.ru