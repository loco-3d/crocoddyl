import unittest
import numpy as np
import cddp


class IntegratorTest(unittest.TestCase):
  def test_forward_Euler(self):
    integrator = cddp.EulerIntegrator()
    discretizer = cddp.EulerDiscretizer()
    dynamics = cddp.SpringMass(integrator, discretizer)
    data = dynamics.createData()

    dt = 1
    x = np.random.rand(dynamics.getConfigurationDimension(), 1)
    u = np.random.rand(dynamics.getControlDimension(), 1)

    x_next = dynamics.stepForward(data, x, u, dt)
    fx, fu = dynamics.computeLinearModel(data, x, u, dt)

    self.assertEqual(x_next.all(), (fx * x + fu * u).all(), \
      "The forward Euler discretizer is wrong.")
  
  def test_RK4(self):
    integrator = cddp.RK4Integrator()
    discretizer = cddp.EulerDiscretizer()
    dynamics = cddp.SpringMass(integrator, discretizer)
    data = dynamics.createData()

    dt = 1
    x = np.random.rand(dynamics.getConfigurationDimension(), 1)
    u = np.random.rand(dynamics.getControlDimension(), 1)

    x_next = dynamics.stepForward(data, x, u, dt)
    fx, fu = dynamics.computeLinearModel(data, x, u, dt)

    self.assertEqual(x_next.all(), (fx * x + fu * u).all(), \
      "The RK4 discretizer is wrong.")

if __name__ == '__main__':
  unittest.main()