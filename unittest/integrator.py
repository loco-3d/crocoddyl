import unittest
import numpy as np
import cddp


class IntegratorTest(unittest.TestCase):
  def test_forward_Euler_integrator(self):
    # Creating the dynamic model of the system and its integrator
    integrator = cddp.EulerIntegrator()
    discretizer = cddp.EulerDiscretizer()
    dynamics = cddp.SpringMass(integrator, discretizer)

    # Creating the dynamics data
    t = 0.
    dt = 1.
    data = dynamics.createData(t, dt)

    # Copying the state and control values
    x = np.random.rand(dynamics.nx(),1)
    u = np.random.rand(dynamics.nu(),1)
    np.copyto(data.x, x)
    np.copyto(data.u, u)
    x_next = np.zeros((dynamics.nx(),1))

    # Integrate the dynamics
    dynamics.updateDynamics(data)
    dynamics.integrator(dynamics, data, x_next)

    # Updating the linear approximation and discretizing it
    dynamics.updateLinearAppr(data)
    dynamics.discretizer(dynamics, data)
    fx = data.discretizer.fx
    fu = data.discretizer.fu

    self.assertEqual(x_next.all(), (np.dot(fx, x) + np.dot(fu, u)).all(), \
      "The forward Euler integrator produces an unexpected x_next vector.")

  # def test_RK4_integrator(self):
  #   # Creating the dynamic model of the system and its integrator
  #   integrator = cddp.RK4Integrator()
  #   discretizer = cddp.EulerDiscretizer()
  #   dynamics = cddp.SpringMass(integrator, discretizer)

  #   # Creating the dynamics data
  #   t = 0.
  #   dt = 1.
  #   data = dynamics.createData(t, dt)

  #   # Copying the state and control values
  #   x = np.random.rand(dynamics.nx(),1)
  #   u = np.random.rand(dynamics.nu(),1)
  #   np.copyto(data.x, x)
  #   np.copyto(data.u, u)
  #   x_next = np.zeros((dynamics.nx(),1))

  #   # Integrate the dynamics
  #   dynamics.updateDynamics(dynamics, data)
  #   dynamics.integrator(dynamics, data, x_next)

  #   # Updating the linear approximation and discretizing it
  #   dynamics.updateLinearAppr(dynamics, data)
  #   dynamics.discretizer(dynamics, data)
  #   fx = data.discretizer.fx
  #   fu = data.discretizer.fu

  #   # x_next = dynamics.stepForward(data, x, u, dt)
  #   # fx, fu = dynamics.computeLinearModel(data, x, u, dt)

  #   self.assertEqual(x_next.all(), (fx * x + fu * u).all(), \
  #     "The RK4 integrator produces an unexpected x_next vector.")

if __name__ == '__main__':
  unittest.main()