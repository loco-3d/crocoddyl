
from cddp.cost_manager import CostManager, np
from cddp.quadratic_cost import *
from cddp.dynamics import DynamicModel


class GoalQuadraticCost(TerminalResidualQuadraticCost):
  def __init__(self, goal):
    k = len(goal)
    TerminalResidualQuadraticCost.__init__(self, k)
    self.x_des = goal

  def r(self, data, x):
    np.copyto(data.r, x - self.x_des)
    return data.r

  def rx(self, data, x):
    np.copyto(data.rx, np.eye(data.n))
    return data.rx


class StateControlQuadraticCost(RunningQuadraticCost):
  def __init__(self, goal):
    RunningQuadraticCost.__init__(self)
    self.x_des = goal

  def xr(self, data, x, u):
    np.copyto(data.xr, x - self.x_des)
    return data.xr

  def ur(self, data, x, u):
    np.copyto(data.ur, u)
    return data.ur


class SpringMass(DynamicModel):
  def __init__(self):
    # State and control dimension
    DynamicModel.__init__(self, 2, 1)

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



# Creating the dynamic model of the system
dynamics = SpringMass()

# Creating the cost manager and its cost functions
x_des = np.matrix([[0.5], [0.]])
cost_manager = CostManager()
goal_cost = GoalQuadraticCost(x_des)
xu_cost = StateControlQuadraticCost(x_des)

# Setting up the weights of the quadratic terms
wx = np.array([100., 100.])
wu = np.array([0.001])
goal_cost.setWeights(0.5 * wx)
xu_cost.setWeights(wx, wu)

# Adding the cost functions to the manager
cost_manager.addTerminal(goal_cost)
cost_manager.addRunning(xu_cost)


# Creating the integrator
from cddp.integrator import *
integrator = EulerIntegrator()

from cddp.ddp_formulation import ConstrainedDDP
timeline = np.arange(0.0, 3., 0.01) #np.linspace(0., 0.5, 51)
ddp = ConstrainedDDP(dynamics, cost_manager, integrator, timeline)
ddp.setInitalState(np.matrix([ [0.1], [0.2] ]))

ddp.init()
ddp.compute()


# l = 0.
# lold = 0.
# for k in range(len(ddp.intervals)-1):
#   it = ddp.intervals[k]
#   data = it.cost
#   l += ddp.cost_manager.computeRunningCost(data, it.x_new, it.u_new)
#   lold += ddp.cost_manager.computeRunningCost(data, it.x, it.u)
  # print '(x,u)', it.x.T, it.u
  # print '(x_new,u_new)', it.x_new.T, it.u_new
#   # # print 'u =', np.asscalar(it.u)
#   # # print 'K =', it.K
#   # # print 'j =', it.j
  # print '---'
  # print

# print l, lold

import cddp.utils as utils
t = timeline[1:-1]
x = np.asarray([np.asscalar(k.x[0]) for k in ddp.intervals[0:-2]])
u = np.asarray([np.asscalar(k.u[0]) for k in ddp.intervals[0:-2]])
utils.plot(t, x)
utils.plot(t, 0.001*u)
utils.show_plot()


