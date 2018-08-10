import numpy as np
import cddp


class SpringMass(cddp.DynamicalSystem):
  """ Spring mass system

  It defines the continuos evolution function (i.e. f(x, u)) and its state and
  control derivatives.
  """
  def __init__(self, integrator, discretizer):
    # State and control dimension
    cddp.DynamicalSystem.__init__(self, 2, 2, 1, integrator, discretizer)

    # Mass, spring and damper values
    self._mass = 10.
    self._stiff = 10.
    self._damping = 5.
    # This is a LTI system, so we can computed onces the A and B matrices
    self._A = np.matrix([[0., 1.],
                         [-self._stiff / self._mass, -self._damping / self._mass]])
    self._B = np.matrix([[0.], [1 / self._mass]])

  def f(self, data, x, u):
    np.copyto(data.f, self._A * x + self._B * u)
    return data.f

  def fx(self, data, x, u):
    np.copyto(data.fx, self._A)
    return data.fx

  def fu(self, data, x, u):
    np.copyto(data.fu, self._B)
    return data.fu

  def advanceConfiguration(self, x, dx):
    return x + dx



class NumDiffSpringMass(cddp.NumDiffDynamicalSystem):
  """ Spring mass system

  It defines the continuos evolution function (i.e. f(x, u)) and computes its state and
  control derivatives through numerical differentiation.
  """
  def __init__(self, integrator, discretizer):
    # State and control dimension
    cddp.NumDiffDynamicalSystem.__init__(self, 2, 2, 1, integrator, discretizer)

    # Mass, spring and damper values
    self._mass = 10.
    self._stiff = 10.
    self._damping = 5.
    # This is a LTI system, so we can computed onces the A and B matrices
    self._A = np.matrix([[0., 1.],
                         [-self._stiff / self._mass, -self._damping / self._mass]])
    self._B = np.matrix([[0.], [1 / self._mass]])

  def f(self, data, x, u):
    np.copyto(data.f, self._A * x + self._B * u)
    return data.f

  def advanceConfiguration(self, x, dx):
    return x + dx