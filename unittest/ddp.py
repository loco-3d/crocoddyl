import unittest
import numpy as np
from cddp.ddp import DDP
from cddp.cost_manager import CostManager
from cddp.integrator import EulerIntegrator
from models.spring_mass import SpringMass
from models.simple_cost import GoalResidualQuadraticCost, StateControlRunningQuadraticCost




plot_enable = False

class LinearDDPTest(unittest.TestCase):
  def setUp(self):
    # Creating the dynamic model of the system
    dynamics = SpringMass()

    # Create random initial and desired state
    x0 = np.random.rand(dynamics.getStateDimension(), 1)
    x_des = np.random.rand(dynamics.getStateDimension(), 1)
    x_des[1] = 0.

    # Creating the cost manager and its cost functions
    cost_manager = CostManager()
    goal_cost = GoalResidualQuadraticCost(x_des)
    xu_cost = StateControlRunningQuadraticCost(x_des)

    # Setting up the weights of the quadratic terms
    wx = np.array([100., 100.])
    wu = np.array([0.001])
    goal_cost.setWeights(0.5 * wx)
    xu_cost.setWeights(wx, wu)

    # Adding the cost functions to the manager
    cost_manager.addTerminal(goal_cost)
    cost_manager.addRunning(xu_cost)

    # Creating the integrator
    integrator = EulerIntegrator()

    # Creating the DDP solver
    timeline = np.arange(0.0, 3., 0.01)  # np.linspace(0., 0.5, 51)
    self.ddp = DDP(dynamics, cost_manager, integrator, timeline)

    # Running the DDP solver
    self.ddp.compute(x0)

    if plot_enable:
      import cddp.utils as utils
      t = timeline[1:-1]
      x = np.asarray([np.asscalar(k.x[0]) for k in self.ddp.intervals[0:-2]])
      u = np.asarray([np.asscalar(k.u[0]) for k in self.ddp.intervals[0:-2]])
      utils.plot(t, x)
      utils.plot(t, 0.001*u)
      utils.show_plot()

  def test_positive_expected_improvement(self):
    self.assertGreater(-self.ddp.dV_exp, 0., "The expected improvement is not positive.")

  def test_positive_obtained_improvement(self):
    self.assertGreater(-self.ddp.dV, 0., "The obtained improvement is not positive.")

  def test_improvement_ratio_equals_one(self):
    self.assertAlmostEqual(np.asscalar(self.ddp.dV) / np.asscalar(self.ddp.dV_exp), 1., 2, "The improvement ration is not equals to 1.")


if __name__ == '__main__':
  unittest.main()
