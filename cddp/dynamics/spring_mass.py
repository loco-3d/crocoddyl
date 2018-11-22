from cddp.dynamics.dynamics import DynamicsModel
from cddp.dynamics.dynamics import DynamicsData
import numpy as np


  
class SpringMassData(DynamicsData):
  def __init__(self, dynamicsModel, dt):
    DynamicsData.__init__(self, dynamicsModel, dt)

    # Mass, spring and damper values
    self._mass = 10.
    self._stiff = 10.
    self._damping = 5.

    # This is a LTI system, so we can define onces aq, av and au
    self.aq[0,0] = -self._stiff / self._mass
    self.av[0,0] = -self._damping / self._mass
    self.au[0,0] = 1 / self._mass


class SpringMass(DynamicsModel):
  def __init__(self, discretizer):
    DynamicsModel.__init__(self, 1, 1, 1, discretizer)

  def createData(dynamicsModel, tInit, dt):
    return SpringMassData(dynamicsModel, dt)

  def updateTerms(dynamicsModel, dynamicsData):
    # We don't need to update the dynamics terms since it's a LTI system
    return

  def updateDynamics(dynamicsModel, dynamicsData):
    # We don't need to update the dynamics since it's a LTI system
    q = dynamicsData.x[:dynamicsModel.nq()]
    v = dynamicsData.x[dynamicsModel.nq():]
    np.copyto(dynamicsData.a,
      dynamicsData.aq * q + dynamicsData.av * v +\
      dynamicsData.au * dynamicsData.u)
    return

  def updateLinearAppr(dynamicsModel, dynamicsData):
    # We don't need to update the linear approximation since it's a LTI system
    return

  def integrateConfiguration(dynamicsModel, dynamicsData, q, dq):
    return q + dq

  def differenceConfiguration(dynamicsModel, dynamicsData, q0, q1):
    return q1 - q0









# from cddp.system import DynamicalSystem, NumDiffDynamicalSystem


# class SpringMass(DynamicalSystem):
#   """ Spring mass system.

#   It defines the continuos evolution function (i.e. f(x,u)) and its state and
#   control derivatives of the spring mass system.
#   """
#   def __init__(self, integrator, discretizer):
#     """ Construct the spring mass system.

#     :param integrator: integration scheme
#     :param discretizer: discretization scheme
#     """
#     # State and control dimension
#     DynamicalSystem.__init__(self, 2, 2, 1, integrator, discretizer)

#     # Mass, spring and damper values
#     self._mass = 10.
#     self._stiff = 10.
#     self._damping = 5.
#     # This is a LTI system, so we can computed onces the A and B matrices
#     self._A = np.matrix([[0., 1.],
#                          [-self._stiff / self._mass, -self._damping / self._mass]])
#     self._B = np.matrix([[0.], [1 / self._mass]])

#   def f(self, data, x, u):
#     """ Compute the spring mass evolution and store it the result in data.

#     :param data: system data
#     :param x: state vector
#     :param u: control input
#     """
#     np.copyto(data.f, self._A * x + self._B * u)
#     return data.f

#   def fx(self, data, x, u):
#     """ Compute the state Jacobian of spring mass system and store it the
#     result in data.

#     :param data: system data
#     :param x: state vector
#     :param u: control input
#     """
#     np.copyto(data.fx, self._A)
#     return data.fx

#   def fu(self, data, x, u):
#     """ Compute the control Jacobian of spring mass system and store it the
#     result in data.

#     :param data: system data
#     :param x: state vector
#     :param u: control input
#     """
#     np.copyto(data.fu, self._B)
#     return data.fu

#   def advanceConfiguration(self, x, dx):
#     """ Operator that advances the configuration state.

#     :param x: state vector
#     :param dx: state vector displacement
#     :returns: the next state value
#     """
#     return x + dx

#   def differenceConfiguration(self, x_next, x_curr):
#     """ Operator that differentiates the configuration state.

#     :param x_next: next state
#     :param x_curr: current state
#     """
#     return x_next - x_curr


# class NumDiffSpringMass(NumDiffDynamicalSystem):
#   """ Spring mass system.

#   It defines the continuos evolution function (i.e. f(x,u)) and computes its
#   state and control derivatives through numerical differentiation of the
#   spring mass system.
#   """
#   def __init__(self, integrator, discretizer):
#     """ Construct the spring mass system.

#     :param integrator: integration scheme
#     :param discretizer: discretization scheme
#     """
#     # State and control dimension
#     NumDiffDynamicalSystem.__init__(self, 2, 2, 1, integrator, discretizer)

#     # Mass, spring and damper values
#     self._mass = 10.
#     self._stiff = 10.
#     self._damping = 5.
#     # This is a LTI system, so we can computed onces the A and B matrices
#     self._A = np.matrix([[0., 1.],
#                          [-self._stiff / self._mass, -self._damping / self._mass]])
#     self._B = np.matrix([[0.], [1 / self._mass]])

#   def f(self, data, x, u):
#     """ Compute the spring mass evolution and store it the result in data.

#     :param data: system data
#     :param x: state vector
#     :param u: control input
#     """
#     np.copyto(data.f, self._A * x + self._B * u)
#     return data.f

#   def advanceConfiguration(self, x, dx):
#     """ Operator that advances the configuration state.

#     :param x: state vector
#     :param dx: state vector displacement
#     :returns: the next state value
#     """
#     return x + dx

#   def differenceConfiguration(self, x_next, x_curr):
#     """ Operator that differentiates the configuration state.

#     :param x_next: next state
#     :param x_curr: current state
#     """
#     return x_next - x_curr
