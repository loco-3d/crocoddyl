import numpy as np
from cddp.dynamics import DynamicModel, NumDiffDynamicModel


class SpringMass(DynamicModel):
  """ Spring mass system

  It defines the continuos evolution function (i.e. f(x, u)) and its state and
  control derivatives.
  """
  def __init__(self, integrator, discretizer):
    # State and control dimension
    DynamicModel.__init__(self, 2, 2, 1, integrator, discretizer)

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


class NumDiffSpringMass(NumDiffDynamicModel):
  """ Spring mass system

  It defines the continuos evolution function (i.e. f(x, u)) and computes its state and
  control derivatives through numerical differentiation.
  """
  def __init__(self):
    # State and control dimension
    NumDiffDynamicModel.__init__(self, 2, 2, 1)

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

  def computePerturbedConfiguration(self, x, index):
    # A perturbation in the tangent manifold has the same effect in the
    # configuration manifold because it's a classical system.
    x_pert = x.copy()
    x_pert[index] += self.sqrt_eps
    return x_pert