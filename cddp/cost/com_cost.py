from cddp.cost import RunningResidualQuadraticCost
import numpy as np
import pinocchio as se3
import math


class CoMRunningCost(RunningResidualQuadraticCost):
  """ Running cost given a desired CoM position.

  This cost function can be only defined for geometric systems. For these 
  systems, the state x=(q,v) contains the configuration point q\in Q and its the
  tangent velocity v\in TxQ. The dimension of the configuration space Q, its
  tangent space TxQ and the control (input) vector are nq, nv and m,
  respectively.
  """
  def __init__(self, model, com_des):
    """ Construct the running cost for a desired SE3.

    It requires the robot model created by Pinocchio and the desired CoM
    position.
    :param model: Pinocchio model
    :com_des: desired CoM position
    """
    self._model = model
    self._data = self._model.createData()
    self.com_des = com_des
    RunningResidualQuadraticCost.__init__(self, 3)

  def r(self, system, data, x, u):
    """ Compute CoM error vector and store the result in data.

    We computed the CoM position given a state x.
    :param system: system
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: SE3 error
    """
    # Computing the actual CoM position
    q = x[:self._model.nq]
    act_com = se3.centerOfMass(self._model, self._data, q)

    # CoM error
    np.copyto(data.r, act_com - self.com_des)
    return data.r

  def rx(self, system, data, x, u):
    """ Compute the state Jacobian of the CoM error vector and store the result
    in data.

    :param system: system
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: state Jacobian of the SE3 error vector
    """
    q = x[:self._model.nq]
    data.rx[:,:self._model.nv] = \
      se3.jacobianCenterOfMass(self._model, self._data, q)
    return data.rx

  def ru(self, system, data, x, u):
    """ Compute the control Jacobian of the CoM error vector and store the result
    in data.

    :param system: system
    :param data: running cost data
    :param x: state vector [joint configuration, joint velocity]
    :param u: control vector
    :returns: control Jacobian of the SE3 error vector
    """
    # This should return an null Jacobian
    return data.ru