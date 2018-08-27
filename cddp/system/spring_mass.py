import numpy as np
from cddp.system import DynamicalSystem, NumDiffDynamicalSystem


class SpringMass(DynamicalSystem):
  """ Spring mass system.

  It defines the continuos evolution function (i.e. f(x,u)) and its state and
  control derivatives of the spring mass system.
  """
  def __init__(self, integrator, discretizer):
    """ Construct the spring mass system.

    :param integrator: integration scheme
    :param discretizer: discretization scheme
    """
    # State and control dimension
    DynamicalSystem.__init__(self, 2, 2, 1, integrator, discretizer)

    # Mass, spring and damper values
    self._mass = 10.
    self._stiff = 10.
    self._damping = 5.
    # This is a LTI system, so we can computed onces the A and B matrices
    self._A = np.matrix([[0., 1.],
                         [-self._stiff / self._mass, -self._damping / self._mass]])
    self._B = np.matrix([[0.], [1 / self._mass]])

  def f(self, data, x, u):
    """ Compute the spring mass evolution and store it the result in data.

    :param data: system data
    :param x: state vector
    :param u: control input
    """
    np.copyto(data.f, self._A * x + self._B * u)
    return data.f

  def fx(self, data, x, u):
    """ Compute the state Jacobian of spring mass system and store it the
    result in data.

    :param data: system data
    :param x: state vector
    :param u: control input
    """
    np.copyto(data.fx, self._A)
    return data.fx

  def fu(self, data, x, u):
    """ Compute the control Jacobian of spring mass system and store it the
    result in data.

    :param data: system data
    :param x: state vector
    :param u: control input
    """
    np.copyto(data.fu, self._B)
    return data.fu

  def advanceConfiguration(self, x, dx):
    """ Operator that advances the configuration state.

    :param x: state vector
    :param dx: state vector displacement
    :returns: the next state value
    """
    return x + dx

  def differentiateConfiguration(self, x_next, x_curr):
    """ Operator that differentiates the configuration state.

    :param x_next: next state
    :param x_curr: current state
    """
    return x_next - x_curr


class NumDiffSpringMass(NumDiffDynamicalSystem):
  """ Spring mass system.

  It defines the continuos evolution function (i.e. f(x,u)) and computes its
  state and control derivatives through numerical differentiation of the
  spring mass system.
  """
  def __init__(self, integrator, discretizer):
    """ Construct the spring mass system.

    :param integrator: integration scheme
    :param discretizer: discretization scheme
    """
    # State and control dimension
    NumDiffDynamicalSystem.__init__(self, 2, 2, 1, integrator, discretizer)

    # Mass, spring and damper values
    self._mass = 10.
    self._stiff = 10.
    self._damping = 5.
    # This is a LTI system, so we can computed onces the A and B matrices
    self._A = np.matrix([[0., 1.],
                         [-self._stiff / self._mass, -self._damping / self._mass]])
    self._B = np.matrix([[0.], [1 / self._mass]])

  def f(self, data, x, u):
    """ Compute the spring mass evolution and store it the result in data.

    :param data: system data
    :param x: state vector
    :param u: control input
    """
    np.copyto(data.f, self._A * x + self._B * u)
    return data.f

  def advanceConfiguration(self, x, dx):
    """ Operator that advances the configuration state.

    :param x: state vector
    :param dx: state vector displacement
    :returns: the next state value
    """
    return x + dx

  def differentiateConfiguration(self, x_next, x_curr):
    """ Operator that differentiates the configuration state.

    :param x_next: next state
    :param x_curr: current state
    """
    return x_next - x_curr