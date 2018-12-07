from cddp.dynamics.dynamics import DynamicsModel
from cddp.dynamics.dynamics import DynamicsData
import numpy as np


class SpringMassData(DynamicsData):
  def __init__(self, dynamicsModel, t, dt):
    DynamicsData.__init__(self, dynamicsModel, t, dt)

    # Mass, spring and damper values
    self._mass = 10.
    self._stiff = 10.
    self._damping = 5.

    # This is a LTI system, so we can define onces aq, av and au
    self.aq[0,0] = -self._stiff / self._mass
    self.av[0,0] = -self._damping / self._mass
    self.au[0,0] = 1 / self._mass


class SpringMass(DynamicsModel):
  def __init__(self, integrator, discretizer):
    DynamicsModel.__init__(self, integrator, discretizer, 1, 1, 1)

  def createData(self, t, dt):
    return SpringMassData(self, t, dt)

  def updateTerms(self, dynamicsData, x):
    # We don't need to update the dynamics terms since it's a LTI system
    return

  def updateDynamics(self, dynamicsData, x, u):
    # We don't need to update the dynamics since it's a LTI system
    q = x[:self.nq()]
    v = x[self.nq():]
    np.copyto(dynamicsData.a,
      dynamicsData.aq * q + dynamicsData.av * v +\
      dynamicsData.au * u)
    return

  def updateLinearAppr(self, dynamicsData, x, u):
    # We don't need to update the linear approximation since it's a LTI system
    return

  def integrateConfiguration(self, q, dq):
    return q + dq

  def differenceConfiguration(self, q0, q1):
    return q1 - q0